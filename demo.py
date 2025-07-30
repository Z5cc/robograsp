import numpy as np

from Env import make_env
from Robot import Robot
from Utilities import Object, Camera



def user_control_demo():

    env = make_env()
    env.reset()
    while True:
        env.step_move(env.read_debug_parameter())








user_control_demo()
