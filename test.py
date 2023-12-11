from toy import Rectangle
from raycast import Raycast
from car import Car
from world import World
from vector import Vector
from utils import draw
from replay_buffer import Transition, ReplayBuffer
from ddpg import DDPG

import torch
import numpy as np

from math import pi
import random
from random import randint
import pygame
import sys



SCREEN_WIDTH = 256
SCREEN_HEIGHT = 256

FRAME = 30
EPISODE = 10000
LEARNING_RATE = 0.0001
GAMMA = 0.99
TAU = 0.0001
BATCH_SIZE = 64

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = DDPG(10, 2, LEARNING_RATE, GAMMA, TAU)
model.load_state_dict(torch.load("automobile_8700.pth"))
model.eval()
buffer = ReplayBuffer(10000)


def get_state(world: World):
    ray_check = world.ray_check()

    res = list(map(lambda r: float(len(r) != 0), ray_check))
    pos = (world.goal.pos - world.car.body.pos)
    pos.rotate(-world.car.direction_rad)
    pos /= SCREEN_WIDTH / 2.
    res += pos.v

    return res


def get_transition(world: World, model: DDPG, random_rate: float):
    state = get_state(world)

    dist1 = (world.car.body.pos - world.goal.pos).norm(2)

    action = model.get_action(torch.FloatTensor(state), random_rate)
    world.car.set_handle(action[0].item() * main_world.car.max_angle)
    world.car.set_speed(action[1].item() * main_world.car.max_speed)
    world.car.update(1/FRAME)

    dist2 = (world.car.body.pos - world.goal.pos).norm(2)

    if len(world.collision_check()) != 0:
        reward = -1.
        next_state = None
        return Transition(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(action).unsqueeze(0), next_state, torch.FloatTensor([reward]).unsqueeze(0))
    elif world.check_goal():
        reward = 1.
        next_state = get_state(world)
        return Transition(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(action).unsqueeze(0), torch.FloatTensor(next_state).unsqueeze(0), torch.FloatTensor([reward]).unsqueeze(0))
    else:
        reward = 0. # (int(dist1 > dist2) - 0.75) / 500
        next_state = get_state(world)
        return Transition(torch.FloatTensor(state).unsqueeze(0), torch.FloatTensor(action).unsqueeze(0), torch.FloatTensor(next_state).unsqueeze(0), torch.FloatTensor([reward]).unsqueeze(0))


def init_world():
    global main_car, goal, main_world
    goal_x = randint(-SCREEN_WIDTH / 2 + 70, SCREEN_WIDTH / 2 - 70)
    goal_y = randint(-SCREEN_HEIGHT / 2 + 70, SCREEN_HEIGHT / 2 - 70)
    
    main_car = Car(
        Rectangle('car', Vector(0, 0), 30, 50, True),
        [
            Raycast(Vector(0, 0), Vector(0, 0), Vector(40 * 1.414, 0)),     # E
            Raycast(Vector(0, 0), Vector(0, 0), Vector(40, 40)),            # NE
            Raycast(Vector(0, 0), Vector(0, 0), Vector(0, 40 * 1.414)),     # N
            Raycast(Vector(0, 0), Vector(0, 0), Vector(-40, 40)),           # NW
            Raycast(Vector(0, 0), Vector(0, 0), Vector(-40 * 1.414, 0)),    # W
            Raycast(Vector(0, 0), Vector(0, 0), Vector(40, -40)),           # SW
            Raycast(Vector(0, 0), Vector(0, 0), Vector(0, -40 * 1.414)),    # S
            Raycast(Vector(0, 0), Vector(0, 0), Vector(-40, -40)),          # SE
        ]
    )
    wall_up = Rectangle('wall', Vector(0, SCREEN_HEIGHT / 2 - 5), SCREEN_WIDTH, 10, True)
    wall_down = Rectangle('wall', Vector(0, -SCREEN_HEIGHT / 2 + 5), SCREEN_WIDTH, 10, True)
    wall_left = Rectangle('wall', Vector(-SCREEN_WIDTH / 2 + 5, 0), 10, SCREEN_HEIGHT, True)
    wall_right = Rectangle('wall', Vector(SCREEN_WIDTH / 2 - 5, 0), 10, SCREEN_HEIGHT, True)
    goal = Rectangle('goal', Vector(goal_x, goal_y), 20, 20, True)
    main_world = World(main_car, goal, [wall_up, wall_down, wall_left, wall_right])


def gui():
    white = (255, 255, 255)
    green = (0, 255, 0)
    red = (255, 0, 0)
    black = (0, 0, 0)

    pygame.init()
    pygame.display.set_caption("Simple PyGame Example")
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

    clock = pygame.time.Clock()

    while True:
        clock.tick(FRAME)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                sys.exit()

        transition = get_transition(main_world, model, 0)
        if transition.reward == 1.:
            goal_x = randint(-SCREEN_WIDTH / 2 + 70, SCREEN_WIDTH / 2 - 70)
            goal_y = randint(-SCREEN_HEIGHT / 2 + 70, SCREEN_HEIGHT / 2 - 70)
            main_world.goal.pos = Vector(goal_x, goal_y)
        if transition.next_state == None:
            init_world()

        key_event = pygame.key.get_pressed()
        if key_event[pygame.K_r]:
            init_world()

        screen.fill(black)
        draw(screen, white, main_world.car.body.pos,  main_world.car.body.points)
        draw(screen, green, main_world.goal.pos, main_world.goal.points)
        
        l = main_world.ray_check()

        for i, ray in enumerate(main_world.car.rays):
            if len(l[i]) == 0:
                draw(screen, green, main_world.car.body.pos, [ray.ray_start, ray.ray_end])
            else:
                draw(screen, red, main_world.car.body.pos, [ray.ray_start, ray.ray_end])
        for toy in main_world.toys:
            draw(screen, white, toy.pos, toy.points)
        pygame.display.update()


init_world()
gui()

