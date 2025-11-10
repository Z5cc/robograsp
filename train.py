from stable_baselines3.common.vec_env import SubprocVecEnv
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
import numpy as np

from DQN import Transition, ReplayMemory, DQN

from Robot import Robot
from Object import Object
from Env import Env





BATCH_SIZE = 1024 # BATCH_SIZE is the number of transitions sampled from the replay buffer
GAMMA = 0.99 # GAMMA is the discount factor as mentioned in the previous section
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.01 # EPS_END is the final value of epsilon
EPS_DECAY = 5000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 0.0003 # LR is the learning rate of the ``AdamW`` optimizer

NUM_EPISODES = 10000
NUM_CPU = 8
C=4
VIS = False







def plot_durations(episode_durations):
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



def select_actions(states): # [V,C,H,W]
    global steps_done
    eps_threshold = EPS_END + (EPS_START - EPS_END) * math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += states.shape[0]

    if random.random() > eps_threshold:
        with torch.no_grad():
            return policy_net(states).max(1).indices # max(1) -> max in dimension 1
    else:
        return torch.tensor([random.randrange(n_actions)for _ in range(NUM_CPU)], device=device, dtype=torch.long) # [V]


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
    state_action_values = policy_net(state_batch).gather(1, action_batch[:,None])

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



def update_states(states, obs, keep=None, reset=None): # tensor[V,C,H,W] array[V,H,W]
    obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
    if keep is not None:
        keep = torch.as_tensor(keep, dtype=torch.bool, device=device)
        if keep.any():
            states = torch.roll(states, shifts=-1, dims=1)
            states[keep,-1] = obs[keep]
    if reset is not None:
        reset = torch.as_tensor(reset, dtype=torch.bool, device=device)
        if reset.any():
            states[reset] = obs[reset][:,None,:,:].expand(-1,C,-1,-1)
    return states





if __name__ == "__main__":
    device = torch.device("cpu")
    vec_env = SubprocVecEnv([lambda: Env(Robot(),Object(),vis=VIS,gamma=GAMMA) for _ in range(NUM_CPU)])
    n_actions = vec_env.action_space.n
    h, w = vec_env.observation_space.shape
    policy_net = DQN(h, w, n_actions).to(device)
    target_net = DQN(h, w ,n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    memory = ReplayMemory(10000)
    steps_done = 0
    episode_durations = []
    episode_durations_cpus = [0]*NUM_CPU


    obs = vec_env.reset() # [V,H,W]
    states = update_states(torch.zeros((NUM_CPU,C,h,w),dtype=torch.float32,device=device), obs, reset=np.ones(NUM_CPU,dtype=bool))

    while len(episode_durations) < NUM_EPISODES:
        
        # 1. RUN ENVIRONMENT AND PUT INTO REPLAY MEMORY
        # 1.1 run environment
        actions = select_actions(states)
        obs, rewards, dones, infos = vec_env.step(actions.tolist())
        # 1.2 process results from run and put into memory
        truncated = np.array([info.get("TimeLimit.truncated") for info in infos], dtype=bool)
        terminated = dones & ~truncated
        obs_truncated = np.array([info.get("terminal_observation", np.zeros((h,w))) for info in infos])
        truncated_states = update_states(states, obs_truncated,truncated)
        next_states = update_states(states, obs, ~dones, dones)
        rewards = torch.from_numpy(rewards).to(device=device)
        for i in range(NUM_CPU):
            s_i = states[i:i+1]
            a_i = actions[i:i+1]
            if truncated[i]:
                ns_i = truncated_states[i:i+1] # truncated states are stopped because of time limit, so naturally they would progress. but next_state already has observation after reset. so need truncatd_states
            elif terminated[i]:
                ns_i = None
            else:
                ns_i = next_states[i:i+1]
            r_i = rewards[i:i+1]
            memory.push(s_i, a_i, ns_i, r_i) # [N,C,H,W] [N] [N,C,H,W] [N] state, action and reward need N dimension for   torch.cat(batch.state). before 'for i' loop N=V, now N=1
        # 1.3 update states
        states = next_states



        # 2. TAKE FROM REPLAY MEMORY AND UPDATE NEURAL NETWORK
        # 2.1 Perform one step of the optimization (on the policy network)
        optimize_model()

        # 2.2 Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        episode_durations_cpus=[t+1 for t in episode_durations_cpus]
        episode_durations = episode_durations + [t for t, done in zip(episode_durations_cpus, dones) if done]
        episode_durations_cpus = [0 if done else t for t, done in zip(episode_durations_cpus, dones)]
        if True in dones:
            plot_durations(episode_durations)
