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
        self.conv1 = nn.Conv2d(1,8,(5,5),padding='same')
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv2 = nn.Conv2d(8,16,(3,3),padding='same')
        self.pool2 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv3 = nn.Conv2d(16,32,(3,3),padding='same')
        self.pool3 = nn.MaxPool2d((2,2),stride=(2,2))
        self.lin1 = nn.Linear(4*4*32, 1000)
        self.lin2 = nn.Linear(1000,300)
        self.lin3 = nn.Linear(300, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.pool3(F.relu(self.conv3(x)))
        x = x.view(x.size(0),-1)
        x = F.relu(self.lin1(x))
        x = F.relu(self.lin2(x))
        return self.lin3(x)
    