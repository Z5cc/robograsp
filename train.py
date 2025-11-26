import matplotlib.pyplot as plt
import torch
from itertools import count

from constants import NUM_EPISODES, DEVICE
from environment.robot import Robot
from environment.obj import Obj
from environment.env import Env
from algorithm.dqnagent import DQNAgent
from algorithm.replaymemory import ReplayMemory
from algorithm.statehandler import StateHandler


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
    plt.pause(0.001)


env = Env()
agent = DQNAgent()
memory = ReplayMemory(10000)
state_handler = StateHandler()
episode_durations = []

for i_episode  in range(NUM_EPISODES):
    obs, info = env.reset()
    state = state_handler.initiate_state(obs)

    for t in count():
        # 1. RUN ENVIRONMENT AND PUT INTO REPLAY MEMORY
        # 1.1 run environment
        action = agent.select_action(state)
        obs, reward, terminated, truncated, info = env.step(action.item())
        # 1.2 process results from run and put into memory
        done = terminated or truncated
        next_state = state_handler.update_state(state, obs)
        reward = torch.tensor([reward]).to(device=DEVICE)
        terminated = torch.tensor([terminated]).to(device=DEVICE)
        memory.push(state, action, next_state, reward, terminated) # [N,C,H,W] [N] [N,C,H,W] [N] [N]
        # 1.3 update state
        state = next_state

        # 2. TAKE FROM REPLAY MEMORY AND UPDATE NEURAL NETWORK
        # 2.1 Perform one step of the optimization on the policy network
        s_batch, a_batch, ns_batch, r_batch, t_batch = memory.sample() # [N,C,H,W] [N] [N,C,H,W] [N] [N]
        if s_batch is not None:
            agent.optimize_model(s_batch, a_batch, ns_batch, r_batch, t_batch)
        # 2.2 Soft update of the target network's weights: θ′ ← τ θ + (1 −τ )θ′
        agent.soft_update()

        if done:
            episode_durations.append(t+1)
            plot_durations(episode_durations)
            break
