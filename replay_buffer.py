from torch.utils.data import Dataset, DataLoader
from torch.utils.data import BatchSampler

from collections import namedtuple, deque
from random import sample


Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.memory = deque([], maxlen=capacity)

    
    def push(self, item: Transition):
        self.memory.append(item)
    

    def sample(self, batch_size: int):
        return sample(self.memory, batch_size)


    def __getitem__(self, idx: int):
        return self.memory[idx]


    def __len__(self):
        return len(self.memory)
