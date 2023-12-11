from vector import Vector

import pygame


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
