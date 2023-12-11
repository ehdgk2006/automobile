from toy import Toy
from car import Car


class World:
    def __init__(self, car: Car, goal: Toy, toys: list[Toy]) -> None:
        self.toys = toys
        self.car = car
        self.goal = goal
        self._length = len(self.toys)
    

    def ray_check(self) -> list[list[str]]:
        res_obj = [[] for _ in range(len(self.car.rays))]

        for i in range(self._length):
            temp = self.car.ray_check(self.toys[i])

            for j in range(len(self.car.rays)):
                if temp[j]:
                    res_obj[j].append(self.toys[i].name)
        
        return res_obj


    def collision_check(self) -> list[str]:
        res = []

        for i in range(self._length):
            if self.car.is_collision(self.toys[i]):
                res.append(self.toys[i].name)
        
        return res
    

    def check_goal(self) -> bool:
        return self.car.is_collision(self.goal)


    def add(self, toy: Toy) -> None:
        self._length += 1
        self.toys.append(toy)
    

    def pop(self, idx: int = -1) -> Toy:
        self._length -= 1
        return self.toys.pop(idx)
    
    
    def remove(self, toy: Toy) -> None:
        self.toys.remove(toy)
        self._length -= 1


    def __getitem__(self, idx: int) -> None:
        return self.toys[idx]



