import random
from copy import deepcopy


class Trajectory:
    def __init__(self, first_state: list[float], first_real_state: list[float]) -> None:
        self.trajectory = []

        self.state = deepcopy(first_state)
        self.real_state = deepcopy(first_real_state)

        self.first_state = deepcopy(first_state)
        self.first_real_state = deepcopy(first_real_state)

        self.score = 0.
        self._length = 0


    def add_trajectory(self, action: list[float], state: list[float], real_state: list[float], reward: float):
        self.trajectory.append(action)
        self.state = state
        self.real_state = real_state
        self.score = reward
        self._length += 1
    

    def reset(self, first_state: list[float], first_real_state: list[float]):
        self.trajectory = []
        self.first_state = first_state
        self.first_real_state = first_real_state
        self._length = 0


    def __eq__(self, __value: object) -> bool:
        if self._length != __value._length:
            return False

        eq_trajectory = True
        for i in range(self._length):
            if self.trajectory[i] != __value.trajectory[i]:
                eq_trajectory = False
                break
        
        if self.score != __value.score:
            return False
        
        eq_state = True
        for i in range(len(self.state)):
            if self.state[i] != __value.state[i]:
                eq_state = False
                break
        
        eq_first_state = True
        for i in range(len(self.state)):
            if self.first_state[i] != __value.first_state[i]:
                eq_first_state = False
                break

        return eq_state and eq_trajectory and eq_first_state
    

    def __len__(self):
        return self._length


class Cell:
    def __init__(self, first_state: list[float], first_real_state: list[float]) -> None:
        self.trajectory = Trajectory(first_state, first_real_state)

        self.times_chosen = 0
        self.times_chosen_since_new = 0
        self.times_seen = 0


    def add_trajectory(self, action: list[float], state: list[float], real_state: list[float], reward: float):
        self.trajectory.add_trajectory(action, state, real_state, reward)

    
    def set_trajectory(self, trajectory: Trajectory):
        self.trajectory = trajectory
    

    def __eq__(self, __value: object) -> bool:
        eq_trajectory = self.trajectory == __value.trajectory
        eq_chosen = self.times_chosen == __value.times_chosen
        eq_chosen_since_new = self.times_chosen_since_new == __value.times_chosen_since_new
        eq_seen = self.times_seen == __value.times_seen

        return eq_trajectory and eq_chosen and eq_chosen_since_new and eq_seen


class Archive:
    def __init__(self, max_len: int = 10000) -> None:
        self.cells = list[Cell]()
        self.max_score = 0.
        self.max_len = max_len
    

    def push(self, cell: Cell):
        has_cell = False
        has_worse_cell = False
        worse_cell_idx = -1

        for i in range(len(self.cells)):
            eq_cell = cell == self.cells[i]
            if eq_cell:
                has_cell = True
            if cell.trajectory.score >= self.cells[i].trajectory.score and cell.trajectory._length <= self.cells[i].trajectory._length and not eq_cell:
                eq_first_state = True
                for j in range(len(self.cells[i].trajectory.first_state)):
                    if cell.trajectory.first_state[j] != self.cells[i].trajectory.first_state[j]:
                        eq_first_state = False
                        break
                eq_state = True
                for j in range(len(self.cells[i].trajectory.state)):
                    if cell.trajectory.state[j] != self.cells[i].trajectory.state[j]:
                        eq_state = False
                        break
                
                if eq_first_state and eq_state:
                    has_worse_cell = True
                    worse_cell_idx = i
        
        if has_worse_cell:
            self.cells[worse_cell_idx] = cell
            self.cells[worse_cell_idx].times_chosen_since_new = 0
        elif not has_cell:
            self.cells.append(cell)
        
        if cell.trajectory.score > self.max_score:
            self.max_score = cell.trajectory.score
        
        if len(self.cells) > self.max_len:
            self.cells = self.cells[1:]


    def chosen_score(self, idx: int, weight: float = 3.0, power: float = 1.0, epsilon1: float = 0.001, epsilon2: float = 0.00001):
        return (weight * ((1 / (self.cells[idx].times_chosen + epsilon1)) ** power)) + epsilon2
    
    
    def chosen_since_new_score(self, idx: int, weight: float = 3.0, power: float = 1.0, epsilon1: float = 0.001, epsilon2: float = 0.00001):
        return (weight * ((1 / (self.cells[idx].times_chosen_since_new + epsilon1)) ** power)) + epsilon2
    
    
    def seen_score(self, idx: int, weight: float = 1.0, power: float = 1.0, epsilon1: float = 0.001, epsilon2: float = 0.00001):
        return (weight * ((1 / (self.cells[idx].times_seen + epsilon1)) ** power)) + epsilon2


    def cnt_score(self, idx: int):
        return self.chosen_score(idx) + self.chosen_since_new_score(idx) + self.seen_score(idx)


    def get_neighbors(self, idx: int):
        return []


    def has_neighber(self, neighber: Cell):
        for cell in self.cells:
            if neighber == cell:
                return True
        
        return False
    

    def neigh_score(self, idx: int, weight: float = 10.0):
        res = 0
        neighbors = self.get_neighbors(idx)

        for neighbor in neighbors:
            res += weight * (1 - int(self.has_neighber(neighbor)))
        
        return res


    def score_weight(self, idx: int):
        return 0.1 ** (self.cells[idx].trajectory.score)


    def cell_score(self, idx: int):
        cnt_score = self.cnt_score(idx)
        neigh_score = self.neigh_score(idx)
        score_weight = self.score_weight(idx)

        return (cnt_score + neigh_score + 1) * score_weight


    def select_cell(self):
        score_sum = 0

        for i in range(len(self.cells)):
            score_sum += self.cell_score(i)
        
        r = random.random()

        for i in range(len(self.cells)):
            cell_probablity = self.cell_score(i) / score_sum

            if r <= cell_probablity:
                self.cells[i].times_chosen += 1
                self.cells[i].times_chosen_since_new += 1
                return self.cells[i]
            else:
                r -= cell_probablity
        
        raise ValueError("Can't select cell")
    

    def see_cell(self, cell: Cell):
        for i in range(len(self.cells)):
            if self.cells[i].trajectory == cell.trajectory:
                self.cells[i].times_seen += 1
                return
        self.push(cell)
