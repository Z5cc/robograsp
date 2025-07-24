import os

import numpy as np
import pybullet as p

from tqdm import tqdm
from Env import Env
from Robot import Robot
from Utilities import Object, Camera
import time
import math


def user_control_demo():
    obj_pos = (0,0,0)

    cam_pos = (0.2, 0.2, 0.2)
    cam_tar = obj_pos
    cam_up_vector = (0, 0, 1)
    near = 0.1 # 0.01 means anything closer than 1 cm is invisible
    far = 5 # anything further than this is also invisible
    size = (48, 48)
    fov = 40

    rob_pos = (0, 0.5, 0)
    rob_orn = (0, 0, 0)
    ll_t = [-0.3,-0.15,0] #x,y,z
    ul_t = [0.3,0.3,0.3]
    ee_center = np.array([0.1,0.1,0.3]) # center for starting position of end effector
    ee_tar = np.array([0,0,0]) # target position for end effector
    cone_tar = np.array(obj_pos) # target position for the restriction cone
    # phi: when taking the z-axis as a vector and rotate it by phi around x, the result is x_c
    # x_c is the center of the restriction cone. x_c is also used as a starting position for the direction of x for the EE
    phi = (np.pi/180)*160
    alpha_l = (np.pi/180)*35 # alpha_l limits alpha for the restriction cone around x_c



    object = Object(obj_pos)
    camera = Camera(cam_pos, cam_tar, cam_up_vector, near, far, size, fov)
    robot = Robot(rob_pos, rob_orn, ll_t, ul_t, ee_center, ee_tar, cone_tar, phi, alpha_l)



    env = Env(robot, object, camera, vis=True)
    env.reset()
    while True:
        obs, reward, done, info = env.step(env.read_debug_parameter())








user_control_demo()
