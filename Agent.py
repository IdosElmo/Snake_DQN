import tensorflow as tf      # Deep Learning library
import numpy as np           # Handle matrices
import SumTree as st
import random
import sys
from collections import deque


class Memory(object):  # stored as ( s, a, r, s_ ) in SumTree
    """
    This SumTree code is modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    """
    PER_e = 0.01  # Hyperparameter that we use to avoid some experiences to have 0 probability of being taken
    PER_a = 0.6  # Hyperparameter that we use to make a tradeoff between taking only exp with high priority and sampling randomly
    PER_b = 0.4  # importance-sampling, from initial value increasing to 1

    PER_b_increment_per_sampling = 0.001

    absolute_error_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        # Making the tree
        """
        Remember that our tree is composed of a sum tree that contains the priority scores at his leaf
        And also a data array
        We don't use deque because it means that at each timestep our experiences change index by one.
        We prefer to use a simple array and to overwrite when the memory is full.
        """
        self.tree = st.SumTree(capacity)

    """
    Store a new experience in our tree
    Each new experience have a score of max_prority (it will be then improved when we use this exp to train our DDQN)
    """

    def store(self, experience):
        # Find the max priority
        max_priority = np.max(self.tree.tree[-self.tree.capacity:])

        # If the max priority = 0 we can't put priority = 0 since this exp will never have a chance to be selected
        # So we use a minimum priority
        if max_priority == 0:
            max_priority = self.absolute_error_upper

        self.tree.add(max_priority, experience)  # set the max p for new p

    """
    - First, to sample a minibatch of k size, the range [0, priority_total] is / into k ranges.
    - Then a value is uniformly sampled from each range
    - We search in the sumtree, the experience where priority score correspond to sample values are retrieved from.
    - Then, we calculate IS weights for each minibatch element
    """

    def sample(self, n):
        # Create a sample array that will contains the minibatch
        memory_b = []

        b_idx, b_ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1), dtype=np.float32)

        # Calculate the priority segment
        # Here, as explained in the paper, we divide the Range[0, ptotal] into n ranges
        priority_segment = self.tree.total_priority / n  # priority segment

        # Here we increasing the PER_b each time we sample a new minibatch
        self.PER_b = np.min([1., self.PER_b + self.PER_b_increment_per_sampling])  # max = 1

        # Calculating the max_weight
        p_min = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_priority
        max_weight = (p_min * n) ** (-self.PER_b)

        for i in range(n):
            """
            A value is uniformly sample from each range
            """
            a, b = priority_segment * i, priority_segment * (i + 1)
            value = np.random.uniform(a, b)

            """
            Experience that correspond to each value is retrieved
            """
            index, priority, data = self.tree.get_leaf(value)

            # P(j)
            sampling_probabilities = priority / self.tree.total_priority

            #  IS = (1/N * 1/P(i))**b /max wi == (N*P(i))**-b  /max wi
            b_ISWeights[i, 0] = np.power(n * sampling_probabilities, -self.PER_b) / max_weight

            b_idx[i] = index

            experience = [data]

            memory_b.append(experience)

        return b_idx, memory_b, b_ISWeights

    """
    Update the priorities on the tree
    """

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.PER_e  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.absolute_error_upper)
        ps = np.power(clipped_errors, self.PER_a)

        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)


# This function helps us to copy one set of variables to another
# In our case we use it when we want to copy the parameters of DQN to Target_network
# Thanks of the very good implementation of Arthur Juliani https://github.com/awjuliani

def update_target_graph():
    # Get the parameters of our DQNNetwork
    from_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "DQNetwork")

    # Get the parameters of our Target_network
    to_vars = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.TRAINABLE_VARIABLES, "TargetNetwork")

    op_holder = []

    # Update our target_network parameters with DQNNetwork parameters
    for from_var, to_var in zip(from_vars, to_vars):
        op_holder.append(to_var.assign(from_var))
    return op_holder


def conv2d_layer(input, channel_in, channel_out, kernel_size, strides, name="conv"):
    with tf.name_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        # print(str(input.shape[0]))
        w = tf.Variable(init(shape=[kernel_size, kernel_size, channel_in, channel_out]), name="W")
        b = tf.Variable(tf.constant(0.0, shape=[channel_out]), name="B")
        conv = tf.nn.conv2d(input, w, strides=[1, strides, strides, 1], padding="VALID")
        act = tf.nn.elu(conv + b)
        tf.compat.v1.summary.histogram("weights", w)
        tf.compat.v1.summary.histogram("biases", b)
        tf.compat.v1.summary.histogram("activations", act)

        return act


def fc_layer(input, channel_in, channel_out, activation=True, name="fc"):
    with tf.name_scope(name):
        init = tf.contrib.layers.xavier_initializer_conv2d()
        w = tf.Variable(init(shape=[channel_in, channel_out]), name="W")
        b = tf.Variable(tf.constant(0.1, shape=[channel_out]), name="B")

        if activation:
            act = tf.nn.elu(tf.matmul(input, w) + b)
            # print(act)
        else:
            act = tf.matmul(input, w) + b
            # print(act)

        tf.compat.v1.summary.histogram("weights", w)
        tf.compat.v1.summary.histogram("biases", b)
        tf.compat.v1.summary.histogram("activations", act)

        return act


class Agent:
    one_hot_actions = [[1, 0], [0, 1]]

    def __init__(self, state_size, action_size, learning_rate, name):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.name = name

        self.K = state_size[0]

        self.stacked_frame = deque([np.zeros([self.K, self.K]) for i in range(4)], maxlen=4)

        # We use tf.variable_scope here to know which network we're using (DQN or target_net)
        # it will be useful when we will update our w- parameters (by copy the DQN parameters)
        with tf.variable_scope(self.name):
            # We create the placeholders
            # *state_size means that we take each elements of state_size in tuple hence is like if we wrote
            # [None, 100, 120, 4]
            self.inputs_ = tf.compat.v1.placeholder(tf.float32, [None, *state_size], name="inputs")

            # image = tf.reshape(self.inputs_, [-1, *state_size])
            # # print(image.shape)
            # tf.compat.v1.summary.image('input', image, 3)

            self.ISWeights_ = tf.compat.v1.placeholder(tf.float32, [None, 1], name='IS_weights')

            self.actions_ = tf.compat.v1.placeholder(tf.float32, [None, action_size], name="actions_")

            # Remember that target_Q is the R(s,a) + ymax Qhat(s', a')
            self.target_Q = tf.compat.v1.placeholder(tf.float32, [None], name="target")

            """
            First convnet:
            CNN
            ELU
            """
            self.conv1 = tf.layers.conv2d(inputs=self.inputs_,
                                          filters=32,
                                          kernel_size=[4, 4],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv1")

            self.conv1_out = tf.nn.elu(self.conv1, name="conv1_out")

            """
            Second convnet:
            CNN
            ELU
            """
            self.conv2 = tf.layers.conv2d(inputs=self.conv1_out,
                                          filters=64,
                                          kernel_size=[2, 2],
                                          strides=[2, 2],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv2")

            self.conv2_out = tf.nn.elu(self.conv2, name="conv2_out")

            """
            Third convnet:
            CNN
            ELU
            """
            self.conv3 = tf.layers.conv2d(inputs=self.conv2_out,
                                          filters=128,
                                          kernel_size=[1, 1],
                                          strides=[1, 1],
                                          padding="VALID",
                                          kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                                          name="conv3")

            self.conv3_out = tf.nn.elu(self.conv3, name="conv3_out")

            self.flatten = tf.layers.flatten(self.conv3_out)

            # Here we separate into two streams
            # The one that calculate V(s)
            self.value_fc = tf.layers.dense(inputs=self.flatten,
                                            units=512,
                                            activation=tf.nn.elu,
                                            kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                            name="value_fc")
            # with tf.variable_scope('value_fc', reuse=tf.AUTO_REUSE):
            #     self.w = tf.get_variable('kernel')

            self.value = tf.layers.dense(inputs=self.value_fc,
                                         units=1,
                                         activation=None,
                                         kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                         name="value")

            # The one that calculate A(s,a)
            self.advantage_fc = tf.layers.dense(inputs=self.flatten,
                                                units=512,
                                                activation=tf.nn.elu,
                                                kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                                name="advantage_fc")

            self.advantage = tf.layers.dense(inputs=self.advantage_fc,
                                             units=self.action_size,
                                             activation=None,
                                             kernel_initializer=tf.contrib.layers.xavier_initializer(),
                                             name="advantages")

            # Agregating layer
            # Q(s,a) = V(s) + (A(s,a) - 1/|A| * sum A(s,a'))
            self.output = self.value + tf.subtract(self.advantage,
                                                   tf.reduce_mean(self.advantage, axis=1, keepdims=True))

            # Q is our predicted Q value.
            self.Q = tf.reduce_sum(tf.multiply(self.output, self.actions_), axis=1)

            # The loss is modified because of PER
            self.absolute_errors = tf.abs(self.target_Q - self.Q)  # for updating Sumtree

            self.loss = tf.reduce_mean(self.ISWeights_ * tf.math.squared_difference(self.target_Q, self.Q))

            self.optimizer = tf.compat.v1.train.AdamOptimizer(self.learning_rate).minimize(self.loss)
            # self.optimizer = tf.compat.v1.train.RMSPropOptimizer(self.learning_rate).minimize(self.loss)

            # d2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.name)
            # # print(d2_vars[0])
            # tf.summary.histogram("weights", d2_vars[0])
            # tf.summary.histogram("biases", d2_vars[1])

            # for var in tf.trainable_variables():
            #     # print(var.eval())
            #     tf.compat.v1.summary.histogram(var.name, var.eval())

    def predict_action(self, explore_start, explore_stop, decay_rate, decay_step, state, sess):
        """
        This function will do the part
        With ϵ select a random action atat, otherwise select at=argmaxaQ(st,a)
        """
        # EPSILON GREEDY STRATEGY
        # Choose action a from state s using epsilon greedy.
        # First we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # Here we'll use an improved version of our epsilon greedy strategy used in Q-learning notebook
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # Make a random action (exploration)
            action = random.randint(0, self.action_size - 1)

        else:
            # Get action from Q-network (exploitation)
            # Estimate the Qs values state
            Qs = sess.run(self.output, feed_dict={self.inputs_: state.reshape((1, *state.shape))})
            # print(Qs)
            # Take the biggest Q value (= the best action)
            action = np.argmax(Qs)

        return action, explore_probability

    def stack_frames(self, state, is_new):
        if is_new:
            self.stacked_frame = deque([np.zeros([self.K, self.K]) for i in range(4)], maxlen=4)

            # s_frames.append(state)
            # s_frames.append(state)
            # s_frames.append(state)
            # s_frames.append(state)

            stack = np.stack(self.stacked_frame, axis=2)
            # print(stack)
        else:
            self.stacked_frame.append(state)

            stack = np.stack(self.stacked_frame, axis=2)
            # print(stack)

        # print(stack.shape)
        return stack
