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
    pos = (0, 0.5, 0)
    orn = (0, 0, 0)
    ll_t = [-0.3,-0.15,0.1] #x,y,z
    ul_t = [0.3,0.3,0.3]
    c = np.array([0.1,0.1,0.3]) # center for starting position of end effector
    # phi: when taking the z-axis as a vector and rotate it by phi around x, the result is x_c
    # x_c is the center of the restriction cone. x_c is also used as a starting position for the direction of x for the EE
    phi = (np.pi/180)*160
    alpha_l = (np.pi/180)*35 # alpha_l limits alpha for the restriction cone around x_c




    object = Object((0,0,0))
    camera = Camera((1, 1, 1),
                    (0, 0, 0),
                    (0, 0, 1),
                    0.1, 5, (320, 320), 40)
    camera = None
    robot = Robot(pos,orn,ll_t,ul_t,c,phi,alpha_l)

    env = Grasping(robot, object, camera, vis=True)
    env.reset()
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter())


if __name__ == '__main__':
    user_control_demo()
