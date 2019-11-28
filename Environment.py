import pygame
import numpy as np
import sys
from collections import deque
import itertools
import math
import time
import random


class World:
    width = 100
    height = 100
    time = 0
    score = 0
    reward = 0
    state_size = 25


class Snake:
    def __init__(self):
        snake_x = World.width * np.random.random()
        snake_y = World.height * np.random.random()

        self.head_position = [int(snake_x), int(snake_y)]
        self.prev_head_position = [int(snake_x), int(snake_y)]

        self.hitbox = 10

        self.snake_positions = deque()
        self.snake_positions.append(self.head_position)
        self.body_length = len(self.snake_positions)

    def update_snake(self, state, prev_head_pos):
        flag = False

        if state == -1:
            if snake.body_length > 1:
                world.reward = snake.body_length
            else:
                world.reward = -1.0
            flag = True

        elif state == 0:
            apple_x = apple.location[0]
            apple_y = apple.location[1]

            p_x = snake.head_position[0]
            p_y = snake.head_position[1]

            dx = np.absolute(apple_x - p_x)
            dy = np.absolute(apple_y - p_y)

            # world.reward = 1 - ((dx ** 2 + dy ** 2) ** 0.5 / world.width  )
            world.reward = 0
            flag = False

        elif state == 1:    # eaten an apple
            # print('???????? ', self.snake_positions, self.head_position, prev_head_pos)
            self.snake_positions[0] = prev_head_pos
            self.snake_positions.appendleft(self.head_position)
            # print('?????? after: ', self.snake_positions, self.head_position, prev_head_pos)
            apple.generate_new_apple()
            world.score += 1
            world.reward = 1.0 * self.body_length
            return False

        for i in range(len(self.snake_positions) - 1, 0, -1):
            if i == 1:
                self.snake_positions[i] = self.prev_head_position
            else:
                self.snake_positions[i] = self.snake_positions[i-1]

        return flag

    # MOVE ONLY THE HEAD OF THE SNAKE
    def move_snake(self, action):
        if action == 0:
            # move snake left
            self.prev_head_position = list(self.head_position)
            # print('one: ', self.prev_head_position, self.head_position)
            snake.head_position[0] -= snake.hitbox
            # print('two: ', self.prev_head_position, self.head_position)

        elif action == 1:
            # move snake up
            self.prev_head_position = list(self.head_position)
            snake.head_position[1] += snake.hitbox

        elif action == 2:
            # move snake right
            self.prev_head_position = list(self.head_position)
            snake.head_position[0] += snake.hitbox

        elif action == 3:
            # move snake down
            self.prev_head_position = list(self.head_position)
            snake.head_position[1] -= snake.hitbox

        # print(self.prev_head_pos, self.head_position)
        collision = check_collision(self.head_position)
        done = self.update_snake(collision, self.prev_head_position)
        self.body_length = len(self.snake_positions)

        return done


class Apple:
    def __init__(self):
        self.location = [0, 0]
        self.generate_new_apple()

    def generate_new_apple(self):
        apple_x = World.width * np.random.random()
        apple_y = World.height * np.random.random()
        self.location = [int(apple_x), int(apple_y)]


def check_collision(position):
    apple_x = int(apple.location[0])
    apple_y = int(apple.location[1])

    p_x = int(position[0])
    p_y = int(position[1])

    dx = np.absolute(apple_x - p_x)
    dy = np.absolute(apple_y - p_y)

    # print([apple_x, apple_y], [p_x, p_y])
    # print(apple.location, position)

    if dx <= snake.hitbox and dy <= snake.hitbox:
        #   AN APPLE HAS BEEN EATEN
        # print([apple_x, apple_y], [p_x, p_y])
        # print(apple.location, position)
        return 1

    elif 0 > p_x or 0 > p_y or p_x > World.width or p_y > World.height:
        #   HIT A waLL
        # print('test2')
        return -1

    elif position in list(itertools.islice(snake.snake_positions, 1, len(snake.snake_positions))):
        # print('test3: ', snake.snake_positions)
        return -1

    elif 0 < p_x < World.width and 0 < p_y < World.height:
        return 0


def start_game():
    global snake, apple, world, window, fps
    snake = Snake()
    apple = Apple()
    world = World()
    apple.generate_new_apple()
    window = pygame.display.set_mode((World.height, World.width))
    pygame.display.set_caption("snake")
    fps = pygame.time.Clock()

    return observe_next()


def game_over():
    pygame.quit()
    sys.exit()


def reset():
    start_game()


def observe_next():

    obs = np.zeros([World.state_size, World.state_size])

    #   SET CENTER POINT
    center = int(world.state_size / 2)
    obs[center: center + 1, center: center + 1] = 1

    apple_loc = apple.location
    current = snake.head_position

    apple_dx = int(np.absolute(apple_loc[0] - current[0]))
    apple_dy = int(np.absolute(apple_loc[1] - current[1]))

    if apple_dx <= center and apple_dy <= center:

        # APPLE IS INSIDE OUR AGENT FRAME
        if apple_loc[0] > current[0] and apple_loc[1] > current[1]:
            # TOP RIGHT
            # print(apple_dx, apple_dy)
            obs[center - apple_dy: center - apple_dy + 1, center + apple_dx: center + apple_dx + 1] = 255.0

        elif apple_loc[0] < current[0] and apple_loc[1] > current[1]:
            # TOP LEFT
            obs[center - apple_dy: center - apple_dy + 1, center - apple_dx: center - apple_dx + 1] = 255.0

        elif apple_loc[0] > current[0] and apple_loc[1] < current[1]:
            # BOT LEFT
            obs[center + apple_dy: center + apple_dy + 1, center - apple_dx: center - apple_dx + 1] = 255.0

        elif apple_loc[0] < current[0] and apple_loc[1] < current[1]:
            # BOT RIGHT
            obs[center + apple_dy: center + apple_dy + 1, center + apple_dx: center + apple_dy + 1] = 255.0

    for pos in snake.snake_positions:

        dx = int(np.absolute(pos[0] - current[0]))
        dy = int(np.absolute(pos[1] - current[1]))

        if pos[0] > current[0] and pos[1] > current[1]:
            # TOP RIGHT
            obs[center - dy: center - dy + 1, center + dx: center + dx + 1] = 1

        elif pos[0] < current[0] and pos[1] > current[1]:
            # TOP LEFT
            obs[center - dy: center - dy + 1, center - dx: center - dx + 1] = 1

        elif pos[0] > current[0] and pos[1] < current[1]:
            # BOT LEFT
            obs[center + dy: center + dy + 1, center - dx: center - dx + 1] = 1

        elif pos[0] < current[0] and pos[1] < current[1]:
            # BOT RIGHT
            obs[center + dy: center + dy + 1, center + dx: center + dx + 1] = 1

    obs = obs.reshape([World.state_size, World.state_size, 1])

    # print(obs.shape)

    return obs


def step(action):

    color0 = (255, 255, 255)
    color1 = (0, 255, 0)
    color2 = (255, 0, 0)

    window.fill(color0)

    done = snake.move_snake(action)

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            game_over()

    pygame.draw.rect(window, color1, pygame.Rect(snake.head_position[0], snake.head_position[1], 10, 10))

    for pos in snake.snake_positions:
        pygame.draw.rect(window, color1, pygame.Rect(pos[0], pos[1], 10, 10))

    pygame.draw.rect(window, color2, pygame.Rect(apple.location[0], apple.location[1], 10, 10))

    next_state = observe_next()
    # print(next_state.shape)
    pygame.display.update()
    fps.tick(60)

    return next_state, world.reward, world.score, done


