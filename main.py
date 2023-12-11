from toy import Toy, Rectangle
from raycast import Raycast
from car import Car
from world import World
from vector import Vector

import pygame
import pygame.color

import sys
import copy
from math import pi


def draw(surface: pygame.Surface, color: tuple, pos: Vector, points: list[Vector]):
    new_points = []

    for point in points:
        new_points.append(Vector(point.v[0], point.v[1]))

        new_points[-1] += pos
        new_points[-1].v[1] *= -1

        width = surface.get_width()
        height = surface.get_height()

        new_points[-1] += Vector((width / 2), (height / 2))

    pygame.draw.lines(surface, color, True, new_points)


SCREEN_WIDTH = 640
SCREEN_HEIGHT = 480
FRAME = 144

main_car = Car(
    Rectangle('car', Vector(0, 0), 30, 50, True),
    [
        Raycast(Vector(0, 0), Vector(0, 0), Vector(50, 50)),
        Raycast(Vector(0, 0), Vector(0, 0), Vector(0, 50 * 1.414)),
        Raycast(Vector(0, 0), Vector(0, 0), Vector(-50, 50)),
        Raycast(Vector(0, 0), Vector(0, 0), Vector(50, -50)),
        Raycast(Vector(0, 0), Vector(0, 0), Vector(0, -50 * 1.414)),
        Raycast(Vector(0, 0), Vector(0, 0), Vector(-50, -50)),
    ]
)
wall_up = Rectangle('wall', Vector(0, SCREEN_HEIGHT / 2 - 5), SCREEN_WIDTH, 10, True)
wall_down = Rectangle('wall', Vector(0, -SCREEN_HEIGHT / 2 + 5), SCREEN_WIDTH, 10, True)
wall_left = Rectangle('wall', Vector(-SCREEN_WIDTH / 2 + 5, 0), 10, SCREEN_HEIGHT, True)
wall_right = Rectangle('wall', Vector(SCREEN_WIDTH / 2 - 5, 0), 10, SCREEN_HEIGHT, True)
goal = Rectangle('goal', Vector(100, 0), 20, 20, True)
main_world = World(main_car, goal, [wall_up, wall_down, wall_left, wall_right])

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

    key_event = pygame.key.get_pressed()
    if key_event[pygame.K_LEFT]:
        main_world.car.set_handle(pi / 4.)
    elif key_event[pygame.K_RIGHT]:
        main_world.car.set_handle(-pi / 4.)
    else:
        main_world.car.set_handle(0)

    if key_event[pygame.K_UP]:
        main_world.car.set_speed(100.)
    elif key_event[pygame.K_DOWN]:
        main_world.car.set_speed(-100.)
    else:
        main_world.car.set_speed(0.)

    main_world.car.update(1/FRAME)

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

