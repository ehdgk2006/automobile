from vector import Vector


class Toy:
    def __init__(self, name: str, pos: Vector, points: list[Vector], collider: bool = False) -> None:
        self.name = name
        self.pos = pos
        self.points = points
        self._length = len(points)
        self.collider = collider

    
    def transform(self, v: Vector) -> None:
        self.pos.transform(v)

    
    def rotate(self, rad: float) -> None:
        for i in range(self._length):
            self.points[i].rotate(rad)
    

    def scale(self, s: float) -> None:
        for i in range(self._length):
            self.points[i].scale(s)


class Polygon(Toy):
    def __init__(self, name: str, pos: Vector, points: list[Vector], collider: bool = False) -> None:
        if len(points) != 3:
            raise ValueError(f"Polygon should have 3 points but be given {len(points)}.")
        
        super().__init__(name, pos, points, collider)


class Rectangle(Toy):
    def __init__(self, name: str, pos: Vector, width: float, height: float, collider: bool = False) -> None:
        pos = pos
        points = [
            Vector(-width / 2., height / 2.),
            Vector(width / 2., height / 2.),
            Vector(width / 2., -height / 2.),
            Vector(-width / 2., -height / 2.),
        ]

        super().__init__(name, pos, points, collider)
        self.width = width
        self.height = height
