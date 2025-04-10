import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
import collections
import torch
import torch.optim as optim
from pybullet_envs.bullet.kuka_diverse_object_gym_env import KukaDiverseObjectEnv
import pybullet as p
import modules.Screen as Screen
from modules.DQN import DQN
from modules.ReplayMemory import ReplayMemory


BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.1
EPS_DECAY = 200
EPS_DECAY_LAST_FRAME = 10**4
TARGET_UPDATE = 1000
LEARNING_RATE = 1e-4
STACK_SIZE = 5
PATH = 'policy_dqn.pt'




def get_env_and_device(isTest):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = KukaDiverseObjectEnv(renders=False, isDiscrete=True, removeHeightHack=False, maxSteps=20, isTest=isTest)
    env.cid = p.connect(p.DIRECT)
    n_actions = env.action_space.n
    return env, device, n_actions

def get_screen_size(env, device):
    init_screen = Screen.get_screen(env, device)
    _, _, screen_height, screen_width = init_screen.shape
    return screen_height, screen_width

def build_train_model():
    env, device, n_actions = get_env_and_device(isTest = False)
    screen_height, screen_width = get_screen_size(env, device)

    policy_net = DQN(screen_height, screen_width, n_actions, STACK_SIZE).to(device)
    target_net = DQN(screen_height, screen_width, n_actions, STACK_SIZE).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()
    optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
    return n_actions, env, device, policy_net, target_net, optimizer

def build_test_model():
    env, device, n_actions = get_env_and_device(isTest = False)
    screen_height, screen_width = get_screen_size(env, device)

    policy_net = DQN(screen_height, screen_width, n_actions, STACK_SIZE).to(device)
    # load the model
    checkpoint = torch.load(PATH)
    policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
    return n_actions, env, device, policy_net
