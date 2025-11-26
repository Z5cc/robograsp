import numpy as np
import pybullet as p


BATCH_SIZE = 128
GAMMA = 0.99 # discount factor
EPS_START = 0.9 # starting value of epsilon
EPS_END = 0.01 # final value of epsilon
EPS_DECAY = 5000 # controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # update rate of the target network
LR = 0.0003 # learning rate
DEVICE = 'cpu'
NUM_EPISODES = 10000

VIS = False # visualization of pybullet
REALTIME = False # running simulation in realtime
SIMULATION_STEP_DELAY = 1 / 240. # one simulation step accounts for that amount of time

C = 4 # for incorporating not only present, but also past images
H = 16 # height of image
W = 16 # width of image
N_ACTIONS = 13 # N_ACTIONS=7 for translational seek, N_ACTIONS=13 for translational and rotational seek

BASE_POS = (0,0.5,0) # robot base position
BASE_ORN = p.getQuaternionFromEuler((0,0,0)) # robot base orientation
TCP_UP = np.array([0,-1,0]) # up vector for tcp
CONE_CENTER = np.array([0,0.05,0.20]) # also center for starting position of tcp
CONE_TAR = np.array([0,0,0]) # target position for the restriction cone
CONE_PHI = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c
LL_T = [-0.15,-0.15,0.03] # lower limit of translation x,y,z
UL_T = [0.15,0.15,0.20] # upper limit of translation x,y,z
NUMERIC_DAMPING = 0.00001 # numeric damping damping value for inverse kinematis (no physical value)
