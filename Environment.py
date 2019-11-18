import pygame
import numpy as np
from collections import deque
import math
import time
import random


class Snake:
    def __init__(self):
        snake_x = world.width * np.random.random()
        snake_y = world.height * np.random.random()
        self.head_position = [snake_x, snake_y]
        self.prev_head_pos = self.head_position

        self.snake_positions = deque()
        self.snake_positions.append(self.head_position)
        self.body_length = len(self.snake_positions)

    def update_snake(self, state):
        for i in range(len(self.snake_positions) - 1, 0, -1):
            self.snake_positions[i] = self.snake_positions[i-1]

        if state < 0:
            return True

        elif state == 0:
            self.snake_positions[0] = self.head_position

        elif state == 1:    # eaten an apple
            self.snake_positions.appendleft(self.head_position)

        return True

    # MOVE ONLY THE HEAD OF THE SNAKE
    def move_snake(self, action):
        if action == 0:
            # move snake left
            self.prev_head_pos = self.head_position
            self.head_position[0] -= 1

        elif action == 1:
            # move snake up
            self.prev_head_pos = self.head_position
            self.head_position[1] += 1

        elif action == 2:
            # move snake right
            self.prev_head_pos = self.head_position
            self.head_position[0] += 1

        elif action == 3:
            # move snake down
            self.prev_head_pos = self.head_position
            self.head_position[1] -= 1

        collision = check_collision(self.head_position)
        done = self.update_snake(collision)

        return done


class Apple:
    def __init__(self):
        self.location = [0, 0]

    def generate_new_apple(self):
        apple_x = world.width * np.random.random()
        apple_y = world.height * np.random.random()
        self.location = [apple_x, apple_y]


class World:
    width = 600
    height = 600
    time = 0
    score = 0


def check_collision(position):
    apple_x = apple.location[0]
    apple_y = apple.location[1]

    p_x = position[0]
    p_y = position[1]

    if p_x == apple_x and p_y == apple_y:
        #   AN APPLE HAS BEEN EATEN
        return 1

    elif 0 < p_x or 0 > p_y or p_x > world.width or p_y > world.height:
        #   HIT A waLL
        return -1

    elif position in snake.snake_positions[1:]:
        return -1

    elif 0 < p_x < world.width and 0 < p_y < world.height:
        return 0


def start_game():
    global snake, apple, world
    snake = Snake()
    apple = Apple()
    world = World()
    apple.generate_new_apple()


def step(action):




