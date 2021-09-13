import random
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'not_done'))


class ReplayBuffer(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""

        # Check if current position lower than replay buffer's size, hence, to avoid cases of out of bound index
        if len(self.memory) < self.capacity:
            self.memory.append(Transition(*args))
            self.position += 1

        # In case of out of bound index, initialize position for overriding past buffers and storing new buffers
        else:
            self.position = self.position % self.capacity
            self.memory[self.position] = Transition(*args)

    def sample(self, batch_size):

        # Sample buffers from replay buffer. Buffer's amount based on the batch size
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
