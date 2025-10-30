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

    def __init__(self, h, w, n_actions):
        super(DQN, self).__init__()
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(1,8,(3,3),padding='same')
        self.conv2 = nn.Conv2d(8,16,(3,3),padding='same')
        self.conv3 = nn.Conv2d(16,16,(3,3),padding='same')
        self.lin1 = nn.Linear(8*8*16, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0),-1)
        x = self.lin1(x)
        return x
