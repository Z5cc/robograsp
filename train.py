import gymnasium as gym
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import deque
from itertools import count
import time


import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from DQN import Transition, ReplayMemory, DQN

from Robot import Robot
from Object import Object
from Env import Env






BATCH_SIZE = 128 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.01 # EPS_END is the final value of epsilon
EPS_DECAY = 5000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 0.0003 # LR is the learning rate of the ``AdamW`` optimizer







# if GPU is to be used
device = torch.device("cpu")
steps_done = 0
episode_durations = []

robot = Robot()
object = Object()
env = Env(robot,object)
h, w = env.camera.HEIGTH, env.camera.WIDTH
n_actions = env.action_space_size
policy_net = DQN(h, w, n_actions).to(device)
target_net = DQN(h, w ,n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)











def plot_durations():
    plt.figure(1)
    durations_t = torch.tensor(episode_durations, dtype=torch.float)
    plt.clf()
    plt.xlabel('Episode')
    plt.ylabel('Duration')
    plt.plot(durations_t.numpy())
    # plot 50 episode average
    if len(durations_t) >= 50:
        means = durations_t.unfold(0, 50, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(49), means))
        plt.plot(means.numpy())
    plt.pause(0.001)  # pause a bit so that plots are updated



def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    # print(f'eps_threshold:{eps_threshold}')
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                  device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)


    # Compute Q(s_t, a)
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1})
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch

    # Compute Huber loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()













if torch.cuda.is_available() or torch.backends.mps.is_available():
    num_episodes = 10000
else:
    num_episodes = 10000

start_tt = time.time()
for i_episode in range(num_episodes):
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0) # [1,1,H,W]
    state = state.repeat(1, 4, 1, 1)  # [1,C,H,W]
    for t in count():
        print(f'\n\n\nt:{t}\n')
        start_t = time.time()
        # 1. RUN ENVIRONMENT
        action = select_action(state)

        start = time.time()
        observation, reward, terminated, truncated, info = env.step(action.item(),GAMMA)
        print(f'simulation step time: {(time.time()-start):.6f}seconds')

        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = state # [1,C,H,W]
            next_state = torch.roll(next_state, shifts=-1, dims=1)
            next_state[:, -1] = torch.as_tensor(observation, dtype=torch.float32, device=device).unsqueeze(0).unsqueeze(0)

        memory.push(state, action, next_state, reward)
        state = next_state


        # 2. UPDATE VALUE FUNCTION -> UPDATE NN
        # Perform one step of the optimization (on the policy network)
        start = time.time()
        optimize_model()
        print(f'optimize model: {(time.time()-start):.6f}seconds')

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        print(f't time: {(time.time()-start_t):.6f}seconds,    total time: {(time.time()-start_tt):.6f}seconds')        
        if done:
            episode_durations.append(t + 1)
            plot_durations()
            break
