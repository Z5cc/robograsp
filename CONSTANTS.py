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
SIMULATION_STEP_DELAY = 1 / 240.

C = 4
H = 16
W = 16
N_ACTIONS = 7
