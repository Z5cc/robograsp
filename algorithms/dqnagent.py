import math
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from CONSTANTS import BATCH_SIZE, GAMMA, EPS_START, EPS_END, EPS_DECAY, TAU, LR, DEVICE, N_ACTIONS


class Network(nn.Module):

    def __init__(self):
        super(Network, self).__init__()
        self.pool1 = nn.MaxPool2d((2,2),stride=(2,2))
        self.conv1 = nn.Conv2d(4,8,(3,3),padding='same')
        self.conv2 = nn.Conv2d(8,16,(3,3),padding='same')
        self.conv3 = nn.Conv2d(16,16,(3,3),padding='same')
        self.lin1 = nn.Linear(8*8*16, N_ACTIONS)

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


class DQNAgent():

    def __init__(self):
        self.policy_net = Network().to(DEVICE)
        self.target_net = Network().to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=LR, amsgrad=True)
        self.steps_done = 0

    def select_action(self, state): # [N,C,H,W]
        eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * self.steps_done / EPS_DECAY)
        self.steps_done += state.shape[0]
        if random.random() > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices # max(1) -> max in dimension 1
        else:
            return torch.tensor([random.randrange(N_ACTIONS)], device=DEVICE, dtype=torch.long) # [N]

    def optimize_model(self, state_batch, action_batch, next_state_batch, reward_batch, terminated_batch):
        # Compute Q(s_t, a)
        action_batch = action_batch[:,None] # add dimension because gather funciton will need it
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute Q(s_t+1, a_max) 
        next_state_values = torch.zeros(BATCH_SIZE, device=DEVICE) # default for 0
        with torch.no_grad():
            next_state_values[~terminated_batch] = self.target_net(next_state_batch[~terminated_batch]).max(1).values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def soft_update(self):
        target_net_state_dict = self.target_net.state_dict()
        policy_net_state_dict = self.policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        self.target_net.load_state_dict(target_net_state_dict)
