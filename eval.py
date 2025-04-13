import collections
from itertools import count
import torch
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet as p
import modules.Screen as Screen
import modules.Config as c



n_actions, env, device, policy_net = c.build_test_model()


episode = 10
scores_window = collections.deque(maxlen=100)  # last 100 scores

# evaluate the model
for i_episode in range(episode):
    env.reset()
    state = Screen.get_screen(env, device)
    stacked_states = collections.deque(c.STACK_SIZE*[state],maxlen=c.STACK_SIZE)
    for t in count():
        stacked_states_t =  torch.cat(tuple(stacked_states),dim=1)
        # Select and perform an action
        action = policy_net(stacked_states_t).max(1)[1].view(1, 1)
        _, reward, done, _ = env.step(action.item())
        # Observe new state
        next_state = Screen.get_screen(env, device)
        stacked_states.append(next_state)
        if done:
            break
    print("Episode: {0:d}, reward: {1}".format(i_episode+1, reward), end="\n")