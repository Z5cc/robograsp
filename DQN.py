from collections import namedtuple, deque
import random
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F







Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))



class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class DQN(nn.Module):

    def __init__(self, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Conv2d(1,3,(3,3),padding='same')
        self.layer2 = nn.Conv2d(3,1,(3,3),padding='same')
        self.layer3 = nn.Linear(48*48,n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = x.view(x.size(0),-1)
        return self.layer3(x)
    