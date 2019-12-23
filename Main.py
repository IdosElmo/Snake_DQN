import tensorflow as tf
import random
import numpy as np
import time
import Environment as Env
from Agent import Agent, Memory, update_target_graph


# MODEL HYPERPARAMETERS
K = 20
num_actions = 4

state_size = [K, K, 1]  # Our input is a stack of 4 frames hence 100x120x4 (Width, height, channels)
action_size = num_actions  # 4 possible actions
one_hot_actions = [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
learning_rate = 0.005  # Alpha (aka learning rate)

# TRAINING HYPERPARAMETERS
total_episodes = 1_000_000  # Total episodes for training
batch_size = 64

# FIXED Q TARGETS HYPERPARAMETERS
max_tau = 10_00    # Tau is the C step where we update our target network

# EXPLORATION HYPERPARAMETERS for epsilon greedy strategy
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.0005  # exponential decay rate for exploration prob

# Q LEARNING hyperparameters
gamma = 0.95  # Discounting rate

# MEMORY HYPERPARAMETERS
# If you have GPU change to 1_million
pretrain_length = 10_0  # Number of experiences stored in the Memory when initialized for the first time
memory_size = 10_0  # Number of experiences the Memory can keep

# MODIFY THIS TO FALSE IF YOU JUST WANT TO SEE THE TRAINED AGENT
training = True
# MODIFY THIS TO TRUE IF YOU WANT TO SEE TRANING IN ACTION
training_render = True
# TURN THIS TO TRUE IF YOU WANT TO RENDER THE ENVIRONMENT
episode_render = True


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAINING ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ PRE POPULATE MEMORY WITH RANDOM ACTIONS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

# Reset the graph
tf.compat.v1.reset_default_graph()

# Instantiate the DQNetwork
DQNetwork = Agent(state_size, action_size, learning_rate, name="DQNetwork")

# Instantiate the target network
TargetNetwork = Agent(state_size, action_size, learning_rate, name="TargetNetwork")

# Saver will help us to save our model
saver = tf.compat.v1.train.Saver()

if training:

    print("training started...\n Pre populating memory.")

    # Instantiate memory
    memory = Memory(memory_size)
    game = Env.Game(500, 20)
    # Render the environment
    state = np.zeros(*[state_size])
    # state = Env.main()
    prev_world_score = 0

    for i in range(pretrain_length):

        _stp = i + 1

        # Random action
        action = random.choice([0, 1, 2, 3])
        # print(action)
        # Make an action within the game
        next_state, reward, score, done = game.step(action, training_render)
        # print(next_state.shape, reward)

        # Look if the episode is finished
        if done:

            # We finished the episode
            # next_state = np.zeros([*state_size])
            # next_state = DQNetwork.observe(None, True)

            # Add experience to memory
            experience = state, one_hot_actions[action], reward, next_state, done
            memory.store(experience)

            # Start a new episode
            game.reset2()
            state = np.zeros(*[state_size])
            # state = Env.start_game()
            prev_world_score = 0

        else:
            # we're not dead

            # Add experience to memory
            # print(reward)
            experience = state, one_hot_actions[action], reward, next_state, done
            memory.store(experience)

            # Our state is now the next_state
            state = next_state
            # print(state.shape)

    print("pre-population finished.")

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ TRAIN FOR NUMBER OF EPISODES USING DDQN Agent ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print("initializing training simulations..")
    with tf.compat.v1.Session() as sess:
        # Initialize the variables
        sess.run(tf.compat.v1.global_variables_initializer())

        # Setup TensorBoard Writer
        writer = tf.compat.v1.summary.FileWriter('./tensorboard/dddqn/4', sess.graph)
        # Losses
        # for var in tf.trainable_variables():
        #     # print(var.eval())
        #     tf.compat.v1.summary.histogram(var.name, var.eval())

        tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)
        # tf.compat.v1.summary.scalar("IS-W", DQNetwork.ISWeights_)
        # tf.compat.v1.summary.scalar("Abs err", DQNetwork.absolute_errors)

        write_op = tf.compat.v1.summary.merge_all()

        # Initialize the decay rate (that will use to reduce epsilon)
        decay_step = 0

        # Set tau = 0
        tau = 0

        # Update the parameters of our TargetNetwork with DQN_weights
        update_target = update_target_graph()
        sess.run(update_target)

        for episode in range(total_episodes):
            # Set step to 0
            step = 0
            # print("epizode: ", episode)
            # Initialize the rewards of t/he episode
            episode_rewards = []

            # Make a new episode and observe the first state
            # state = Env.start_game()
            game.reset2()
            state = np.zeros(*[state_size])
            # print(state)
            done = False
            prev_world_score = 0
            start = time.time()

            while not done:

                # Increase the C step
                tau += 1

                # Increase decay_step
                decay_step += 1

                # With ϵ select a random action atat, otherwise select a = argmaxQ(st,a)
                action, explore_probability = Agent.predict_action(DQNetwork, explore_start,
                                                                    explore_stop, decay_rate,
                                                                    decay_step, state, sess)

                # print(state)

                # image = tf.reshape(state, [-1, *state_size])
                # # # print(image.shape)
                # tf.compat.v1.summary.image('input', image, 3)

                # print(DQNetwork.Q.eval())
                # print(DQNetwork.output.eval())

                # Make an action within the game
                next_state, reward, score, done = game.step(action, training_render)
                # print(next_state, reward, score, done, action)
                # Add the reward to total reward
                episode_rewards.append(reward)

                # If the game is finished
                if done:
                    # We finished the episode
                    # # next_state = np.zeros([*state_size])
                    # next_state = DQNetwork.observe(None, True)

                    # Get the total reward of the episode
                    total_reward = np.sum(episode_rewards)

                    print('Episode: {}'.format(episode),
                          'Total reward: {}'.format(total_reward),
                          'Snake Length: {}'.format(score),
                          'Training loss: {:.4f}'.format(loss),
                          'Explore P: {:.4f}'.format(explore_probability),
                          'Time (sec): {:.4f}'.format((time.time() - start)))

                    # Add experience to memory
                    experience = state, one_hot_actions[action], reward, next_state, done
                    memory.store(experience)

                else:
                    # Add experience to memory
                    experience = state, one_hot_actions[action], reward, next_state, done
                    memory.store(experience)
                    # print('state: ', state, '\nnext state: ', next_state)
                    # st+1 is now our current state
                    state = next_state

                # LEARNING PART
                # Obtain random mini-batch from memory
                tree_idx, batch, ISWeights_mb = memory.sample(batch_size)

                states_mb = np.array([each[0][0] for each in batch], ndmin=3)
                actions_mb = np.array([each[0][1] for each in batch])
                rewards_mb = np.array([each[0][2] for each in batch])
                next_states_mb = np.array([each[0][3] for each in batch], ndmin=3)
                dones_mb = np.array([each[0][4] for each in batch])

                # print(actions_mb)
                # print(rewards_mb)
                target_Qs_batch = []

                # DOUBLE DQN Logic
                # Use DQNNetwork to select the action to take at next_state (a') (action with the highest Q-value)
                # Use TargetNetwork to calculate the Q_val of Q(s',a')

                # Get Q values for next_state
                q_next_state = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: next_states_mb})

                # Calculate Qtarget for all actions that state
                q_target_next_state = sess.run(TargetNetwork.output,
                                               feed_dict={TargetNetwork.inputs_: next_states_mb})

                # Set Q_target = r if the episode ends at s+1, otherwise set Q_target = r + gamma * Qtarget(s',a')
                for i in range(0, len(batch)):
                    terminal = dones_mb[i]

                    # We got a'
                    action = np.argmax(q_next_state[i])

                    # If we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards_mb[i])

                    else:
                        # Take the Qtarget for action a'
                        target = rewards_mb[i] + gamma * q_target_next_state[i][action]
                        target_Qs_batch.append(target)

                targets_mb = np.array([each for each in target_Qs_batch])

                # targets_mb = targets_mb.reshape(targets_mb.shape[0])
                # print("tree idx: ", tree_idx)
                # print("is weights: ", ISWeights_mb)
                # print("target: ", targets_mb)

                # print(states_mb.shape, targets_mb.shape, actions_mb.shape, ISWeights_mb.shape)
                # print('states: ', states_mb)
                # print('target: ', targets_mb)
                # print('actions: ', actions_mb)
                # print('Weights: ', ISWeights_mb)

                # print("wtf")

                _, loss, absolute_errors = sess.run(
                    [DQNetwork.optimizer, DQNetwork.loss, DQNetwork.absolute_errors],
                    feed_dict={DQNetwork.inputs_: states_mb,
                               DQNetwork.target_Q: targets_mb,
                               DQNetwork.actions_: actions_mb,
                               DQNetwork.ISWeights_: ISWeights_mb})

                # Update priority
                memory.batch_update(tree_idx, absolute_errors)
                # print(states_mb.shape)
                # Write TF Summaries
                summary = sess.run(write_op, feed_dict={DQNetwork.inputs_: states_mb,
                                                        DQNetwork.target_Q: targets_mb,
                                                        DQNetwork.actions_: actions_mb,
                                                        DQNetwork.ISWeights_: ISWeights_mb,
                                                        TargetNetwork.inputs_: states_mb})

                # conv1_kernel_val = sess.graph.get_tensor_by_name('DQNetwork/conv1/kernel:0').eval()
                # # print(conv1_kernel_val)
                # tf.compat.v1.summary.scalar("kernele", conv1_kernel_val)

                # d2_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, DQNetwork.name)
                # # print(d2_vars[0])
                # tf.summary.histogram("weights", d2_vars[0])
                # tf.summary.histogram("biases", d2_vars[1])

                # gr = sess.graph
                # for op in gr.get_operations():
                #     print(op.name)

                # conv1_kernel_val = gr.get_tensor_by_name('DQNetwork/conv1/kernel:0').eval()
                # conv1_bias_val = gr.get_tensor_by_name('DQNetwork/conv1/bias:0').eval()
                # tf.summary.histogram("weights", conv1_kernel_val)
                # tf.summary.histogram("biases", conv1_bias_val)
                # for var in tf.trainable_variables():
                #     tf.summary.histogram(var.name, var)

                writer.add_summary(summary, episode)
                writer.flush()

                if tau > max_tau:
                    # Update the parameters of our TargetNetwork with DQN_weights
                    update_target = update_target_graph()
                    sess.run(update_target)
                    tau = 0
                    print("Model updated")

            # Save model every 5 episodes
            if episode % 5 == 0:
                save_path = saver.save(sess, './models/model.ckpt')
                print("Model Saved")

    # writer.close()

else:

    with tf.compat.v1.Session() as sess:

        # Load the model
        saver.restore(sess, "./models/model.ckpt")

        writer = tf.compat.v1.summary.FileWriter('./tensorboard/dddqn/4', sess.graph)
        # Losses
        # for var in tf.trainable_variables():
        #     # print(var.eval())
        #     tf.compat.v1.summary.histogram(var.name, var.eval())

        # tf.compat.v1.summary.scalar("Loss", DQNetwork.loss)
        # tf.compat.v1.summary.scalar("IS-W", DQNetwork.ISWeights_)
        # tf.compat.v1.summary.scalar("Abs err", DQNetwork.absolute_errors)

        write_op = tf.compat.v1.summary.merge_all()

        # play for 100 games
        for i in range(100):
            # Start the game
            # state = Env.start_game()
            game = Env.Game(500, 20)
            state = np.zeros(*[state_size])
            done = False
            score = 0
            step = 0

            while not done:
                step += 1

                # EPSILON GREEDY STRATEGY
                # Choose action a from state s using epsilon greedy.
                # First we randomize a number
                exp_exp_tradeoff = np.random.rand()

                explore_probability = 0.01

                if explore_probability > exp_exp_tradeoff:
                    # Make a random action (exploration)
                    action = random.choice([0, 1, 2, 3])

                else:
                    # Get action from Q-network (exploitation)
                    # Estimate the Qs values state
                    Qs = sess.run(DQNetwork.output, feed_dict={DQNetwork.inputs_: state.reshape((1, *state.shape))})
                    action = np.argmax(Qs)

                next_state, score, reward, done = game.step(action)

                if done:
                    break

                else:
                    state = next_state

            print("Score: ", score)
