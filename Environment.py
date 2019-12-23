# Snake Tutorial Python

import math
import random
import numpy as np
import sys
import pygame
import tkinter as tk
from tkinter import messagebox


class cube(object):
    rows = 20
    w = 500
    pos = 0

    def __init__(self, start, dirnx=1, dirny=0, color=(255, 0, 0)):
        self.pos = start
        self.dirnx = 1
        self.dirny = 0
        self.color = color

    def move(self, dirnx, dirny):
        self.dirnx = dirnx
        self.dirny = dirny
        self.pos = (self.pos[0] + self.dirnx, self.pos[1] + self.dirny)

    def draw(self, surface, eyes=False):
        dis = self.w // self.rows
        i = self.pos[0]
        j = self.pos[1]

        pygame.draw.rect(surface, self.color, (i * dis + 1, j * dis + 1, dis - 2, dis - 2))
        if eyes:
            centre = dis // 2
            radius = 3
            circleMiddle = (i * dis + centre - radius, j * dis + 8)
            circleMiddle2 = (i * dis + dis - radius * 2, j * dis + 8)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle, radius)
            pygame.draw.circle(surface, (0, 0, 0), circleMiddle2, radius)


class snake(object):
    body = []
    turns = {}
    hit_wall = False
    done = False

    def __init__(self, color, pos):
        self.done = False
        self.color = color
        self.head = cube(pos)
        self.body.append(self.head)
        self.dirnx = 0
        self.dirny = 1
        self.hit_wall = False

    def move(self, action):
        # pygame.event.
        flag = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()

            # keys = pygame.key.get_pressed()
            #
            # for key in keys:
            #     if keys[pygame.K_LEFT]:
            #         self.dirnx = -1
            #         self.dirny = 0
            #         self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            #         flag = True
            #
            #     elif keys[pygame.K_RIGHT]:
            #         self.dirnx = 1
            #         self.dirny = 0
            #         self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            #         flag = True
            #
            #     elif keys[pygame.K_UP]:
            #         self.dirnx = 0
            #         self.dirny = -1
            #         self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            #         flag = True
            #
            #     elif keys[pygame.K_DOWN]:
            #         self.dirnx = 0
            #         self.dirny = 1
            #         self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            #         flag = True

        if action == 0:
            # print("moved left")
            self.dirnx = -1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            # flag = True

        elif action == 1:
            # print("moved right")
            self.dirnx = 1
            self.dirny = 0
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            # flag = True

        elif action == 2:
            # print("moved up")
            self.dirnx = 0
            self.dirny = -1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            # flag = True

        elif action == 3:
            # print("moved down")
            self.dirnx = 0
            self.dirny = 1
            self.turns[self.head.pos[:]] = [self.dirnx, self.dirny]
            # flag = True

        self.hit_wall = False

        for i, c in enumerate(self.body):
            # print('c: ', c.pos)
            p = c.pos[:]
            # if flag:
            if p in self.turns:
                # print("test222")
                turn = self.turns[p]
                c.move(turn[0], turn[1])
                if i == len(self.body) - 1:
                    self.turns.pop(p)
            else:
                c.move(c.dirnx, c.dirny)
                self.hit_wall = False

            if c.dirnx == -1 and c.pos[0] <= 0:
                # print("hit wall0")
                # c.pos = (c.rows - 1, c.pos[1])
                self.hit_wall = True
            elif c.dirnx == 1 and c.pos[0] >= c.rows - 1:
                # print("hit wall1")
                # c.pos = (0, c.pos[1])
                self.hit_wall = True
            elif c.dirny == 1 and c.pos[1] >= c.rows - 1:
                # print("hit wall2")
                # c.pos = (c.pos[0], 0)
                self.hit_wall = True
            elif c.dirny == -1 and c.pos[1] <= 0:
                # print("hit wall3")
                # c.pos = (c.pos[0], c.rows - 1)
                self.hit_wall = True
            # else:
            #     c.move(c.dirnx, c.dirny)
            #     self.hit_wall = False

    def reset(self, pos):
        self.done = False
        self.head = cube(pos)
        self.body = []
        self.body.append(self.head)
        self.turns = {}
        self.dirnx = 0
        self.dirny = 1

    def addCube(self):
        tail = self.body[-1]
        dx, dy = tail.dirnx, tail.dirny

        if dx == 1 and dy == 0:
            self.body.append(cube((tail.pos[0] - 1, tail.pos[1])))
        elif dx == -1 and dy == 0:
            self.body.append(cube((tail.pos[0] + 1, tail.pos[1])))
        elif dx == 0 and dy == 1:
            self.body.append(cube((tail.pos[0], tail.pos[1] - 1)))
        elif dx == 0 and dy == -1:
            self.body.append(cube((tail.pos[0], tail.pos[1] + 1)))

        self.body[-1].dirnx = dx
        self.body[-1].dirny = dy

    def draw(self, surface):
        for i, c in enumerate(self.body):
            if i == 0:
                c.draw(surface, True)
            else:
                c.draw(surface)


def drawGrid(w, rows, surface):
    sizeBtwn = w // rows

    x = 0
    y = 0
    for l in range(rows):
        x = x + sizeBtwn
        y = y + sizeBtwn

        pygame.draw.line(surface, (255, 255, 255), (x, 0), (x, w))
        pygame.draw.line(surface, (255, 255, 255), (0, y), (w, y))


def randomSnack(rows, item):
    positions = item.body

    while True:
        x = random.randrange(rows)
        y = random.randrange(rows)
        if len(list(filter(lambda z: z.pos == (x, y), positions))) > 0:
            continue
        else:
            break

    return (x, y)


def message_box(subject, content):
    root = tk.Tk()
    root.attributes("-topmost", True)
    root.withdraw()
    messagebox.showinfo(subject, content)
    try:
        root.destroy()
    except:
        pass


class Game:
    width = 500
    prev_dis = 0
    rows = 20

    def __init__(self, width=500, rows=20):
        # global width, rows, s, snack, win
        self.width = width
        self.rows = rows
        self.win = pygame.display.set_mode((width, width))
        self.s = snake((255, 0, 0), (10, 10))
        self.snack = cube(randomSnack(self.rows, self.s), color=(0, 255, 0))
        self.prev_dis = rows

    def redrawWindow(self, surface):
        surface.fill((0, 0, 0))
        self.s.draw(surface)
        self.snack.draw(surface)
        drawGrid(self.width, self.rows, surface)
        pygame.display.update()

    def reset2(self):
        # global width, rows, s, snack, win
        self.width = 500
        self.rows = 20
        self.win = pygame.display.set_mode((self.width, self.width))
        self.s = snake((255, 0, 0), (10, 10))
        self.snack = cube(randomSnack(self.rows, self.s), color=(0, 255, 0))

        self.s.done = False
        self.s.head = cube((10, 10))
        self.s.body = []
        self.s.body.append(self.s.head)
        self.s.turns = {}
        self.s.dirnx = 0
        self.s.dirny = 1

    def calculate_reward(self, head, apple, prev_d):
        # print(head, apple)

        snake_x = head[0]
        snake_y = head[1]

        apple_x = apple[0]
        apple_y = apple[1]

        distance = ((snake_x - apple_x) ** 2 + (snake_y - apple_y) ** 2) ** 0.5

        self.prev_dis = distance

        if distance == 0: distance = 1

        if distance < prev_d:
            reward = 1 / distance

        else:
            reward = 0

        return reward

    def step(self, action, render=False):
        clock = pygame.time.Clock()
        self.s.done = False
        pygame.time.delay(50)
        clock.tick(30)
        self.s.move(action)

        reward = 0

        if self.s.body[0].pos == self.snack.pos:
            self.s.addCube()
            reward = 1 * len(self.s.body)
            self.snack = cube(randomSnack(self.rows, self.s), color=(0, 255, 0))

        # x2 = pygame.surfarray.array3d(self.win)

        state = np.zeros((self.rows, self.rows))

        for i in self.s.body:
            # print(i.pos)
            x = i.pos[0]
            y = i.pos[1]
            state[x][y] = 1

        state[self.snack.pos[0]][self.snack.pos[1]] = 1

        # print(s.head.pos)
        # reward = self.calculate_reward(self.s.head.pos, self.snack.pos, self.prev_dis)

        for x in range(len(self.s.body)):
            if self.s.hit_wall or self.s.body[x].pos in list(map(lambda z: z.pos, self.s.body[x + 1:])):
                print('Score: ', len(self.s.body))

                self.s.done = True
                reward = -1
                break

        self.redrawWindow(self.win)
        # if render:
        #     win.fill((0, 0, 0))
        #     s.draw(win)
        #     snack.draw(win)
        #     drawGrid(width, rows, win)
        #     pygame.display.update()

        score = len(self.s.body)
        return state.reshape((self.rows, self.rows, 1)), reward, score, self.s.done


# main()
#
# run = True
#
# while run:
#     action = random.randrange(4)
#     vector, done = step(action)
#
#     # print(action)
#
#     if done:
#         s.reset((10, 10))
#         # run = False
#
# print("episode finished")

