from math import pi

from vector import Vector
from toy import Toy, Rectangle
from raycast import Raycast


class Car:
    def __init__(self, body: Rectangle, rays: list[Raycast]) -> None:
        self.body = body

        self.body_collider = [
            Raycast(self.body.pos.copy(), self.body.points[0].copy(), self.body.points[1].copy()),
            Raycast(self.body.pos.copy(), self.body.points[1].copy(), self.body.points[2].copy()),
            Raycast(self.body.pos.copy(), self.body.points[2].copy(), self.body.points[3].copy()),
            Raycast(self.body.pos.copy(), self.body.points[3].copy(), self.body.points[0].copy()),
        ]
        self.rays = rays

        self.max_speed = 100.0
        self.max_angle = pi / 4.

        self.speed = 0.0
        self.direction = Vector(0, 1)
        self.direction_rad = 0.0
        self.angle = 0.0
    

    def ray_check(self, toy: Toy) -> list[bool]:
        res = []
        
        if not toy.collider:
            return [False for _ in range(len(self.rays))]
        
        for i in range(len(self.rays)):
            res.append(self.rays[i].is_collision(toy))
        
        return res
    

    def is_collision(self, toy: Toy) -> bool:
        if not toy.collider:
            return False

        for i in range(len(self.body_collider)):
            if self.body_collider[i].is_collision(toy):
                return True
            
        return False


    def transform(self, v: Vector) -> None:
        self.body.transform(v)
        for i in range(len(self.rays)):
            self.rays[i].pos.transform(v)
        
        for i in range(len(self.body_collider)):
            self.body_collider[i].pos.transform(v)
    

    def rotate(self, rad: float) -> None:
        self.body.rotate(rad)
        self.direction.rotate(rad)
        self.direction_rad += rad

        while self.direction_rad >= 2 * pi:
            self.direction_rad -= 2*pi

        for i in range(len(self.rays)):
            self.rays[i].ray_start.rotate(rad)
            self.rays[i].ray_end.rotate(rad)
            
        for i in range(len(self.body_collider)):
            self.body_collider[i].ray_start.rotate(rad)
            self.body_collider[i].ray_end.rotate(rad)
    

    def set_speed(self, s: float):
        if s < -self.max_speed:
            self.speed = -self.max_speed
        elif s > self.max_speed:
            self.speed = self.max_speed
        else:
            self.speed = s
    

    def set_handle(self, rad: float):
        if rad < -self.max_angle:
            self.angle = -self.max_angle
        elif rad > self.max_angle:
            self.angle = self.max_angle
        else:
            self.angle = rad


    def update(self, dt: float):
        self.rotate(self.angle * self.speed * 1/self.body.width * dt)
        self.transform(self.direction * self.speed * dt)
