import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from env import Grasping
from robot import Robot
from utilities import Object, Camera
import time
import math


def user_control_demo():
    object = Object()
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    robot = Robot((0, 0.5, 0), (0, 0, 0))

    env = Grasping(robot, object, camera, vis=True)
    env.reset()
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter())


if __name__ == '__main__':
    user_control_demo()
