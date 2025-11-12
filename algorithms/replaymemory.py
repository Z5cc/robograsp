from collections import namedtuple, deque
import random
import torch

from CONSTANTS import BATCH_SIZE


Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'terminated'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self):
        if len(self.memory) < BATCH_SIZE:
            return None, None, None, None, None
        
        transitions = random.sample(self.memory, BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)
        terminated_batch = torch.cat(batch.terminated)
        return state_batch, action_batch, next_state_batch, reward_batch, terminated_batch

    def __len__(self):
        return len(self.memory)
    