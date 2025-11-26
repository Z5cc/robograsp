import numpy as np
import pybullet as p


BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 0.9 # EPS_START is the starting value of epsilon
EPS_END = 0.01 # EPS_END is the final value of epsilon
EPS_DECAY = 5000 # EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
TAU = 0.005 # TAU is the update rate of the target network
LR = 0.0003
DEVICE = 'cpu'
NUM_EPISODES = 10000

VIS = False
REALTIME = False
SIMULATION_STEP_DELAY = 1 / 240.

C = 4
H = 16
W = 16
N_ACTIONS = 13 # N_ACTIONS=7 for only translational seek movements

BASE_POS = (0,0.5,0) # robot base position
BASE_ORN = p.getQuaternionFromEuler((0,0,0)) # robot base orientation
TCP_UP = np.array([0,-1,0]) # up vector for tcp
CONE_CENTER = np.array([0,0.05,0.20]) # also center for starting position of tcp
CONE_TAR = np.array([0,0,0]) # target position for the restriction cone
CONE_PHI = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c
LL_T = [-0.15,-0.15,0.03] # lower limit of translation x,y,z
UL_T = [0.15,0.15,0.20] # upper limit of translation x,y,z
NUMERIC_DAMPING = 0.00001 # numeric damping damping value for inverse kinematis (no physical value)
