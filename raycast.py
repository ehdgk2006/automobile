from toy import Toy
from vector import Vector


class Raycast:
    def __init__(self, pos: Vector, start: Vector, end: Vector) -> None:
        """
        Initialize Raycast object:
            Parameters:
                pos: position of line, which is origin point
                start: start point of line
                end: end point of line
        """
        self.pos = pos
        self.ray_start = start
        self.ray_end = end

    
    def line_collision(self, pos: Vector, p1: Vector, p2: Vector) -> bool:
        """
        Check Line Collision:
            Parameters:
                pos1: position of line, which is origin point
                p1: start point of line
                p2: end point of line
        """
        np1, np2, np3, np4 = self.ray_start + self.pos, self.ray_end + self.pos, p1 + pos, p2 + pos

        x1, x2, x3, x4 = np1[0], np2[0], np3[0], np4[0]
        y1, y2, y3, y4 = np1[1], np2[1], np3[1], np4[1]

        denominator= ((x2 - x1) * (y4 - y3)) - ((y2 - y1) * (x4 - x3))
        numerator1 = ((y1 - y3) * (x4 - x3)) - ((x1 - x3) * (y4 - y3))
        numerator2 = ((y1 - y3) * (x2 - x1)) - ((x1 - x3) * (y2 - y1))

        if (denominator == 0):
            return (numerator1 == 0 and numerator2 == 0)

        r = numerator1 / denominator
        s = numerator2 / denominator

        return ((r >= 0 and r <= 1) and (s >= 0 and s <= 1))


    def is_collision(self, toy: Toy) -> bool:
        """
        Check Object Collision:
            Parameters:
                toy: Game Object which has points
        """

        if not toy.collider:
            return False

        for i in range(len(toy.points) - 1):
            is_coll = self.line_collision(toy.pos, toy.points[i], toy.points[i+1])

            if is_coll:
                return True
        return self.line_collision(toy.pos, toy.points[0], toy.points[-1])
