import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from Utilities import Object, Camera
from Robot import Robot
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class Env:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, object, camera=None, vis=False) -> None:
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera
        self.object = object
        self.robot = robot

        # load
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        self.object.load()
        self.robot.load()
        self.robot.step_simulation = self.step_simulation

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.dxin = p.addUserDebugParameter("dx", -0.1, 0.1, 0)
        self.dyin = p.addUserDebugParameter("dy", -0.1, 0.1, 0)
        self.dzin = p.addUserDebugParameter("dz", -0.1, 0.1, 0)
        self.drollId = p.addUserDebugParameter("droll", -0.5, 0.5, 0)
        self.dpitchId = p.addUserDebugParameter("dpitch", -0.5, 0.5, 0)
        self.dyawId = p.addUserDebugParameter("dyaw", -0.5, 0.5, 0)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        self.action_space_size = 7
        self.steps = 0
        self.max_steps = 500

    def read_debug_parameter(self):
        # read the value of task parameter
        dx = p.readUserDebugParameter(self.dxin)
        dy = p.readUserDebugParameter(self.dyin)
        dz = p.readUserDebugParameter(self.dzin)
        droll = p.readUserDebugParameter(self.drollId)
        dpitch = p.readUserDebugParameter(self.dpitchId)
        dyaw = p.readUserDebugParameter(self.dyawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return dx, dy, dz, droll, dpitch, dyaw, gripper_opening_length

        






    def step(self, action):
        gr_l = 0.05
        dx = +0.005
        dy,dz = 0,0
        droll,dpitch,dyaw=0,0,0

        if action==0: # grasp
            gr_l = 0
            dx = -0.5
        elif action==1: # approach
            dx = 0.015
        elif action==2: # regrasp
            dx = -0.06
        elif action==3: # adjust while moving little to object
            dy = 0.01
        elif action==4:
            dy = -0.01
        elif action==5:
            dz = +0.01
        elif action==6:
            dz = -0.01

        delta = [dx,dy,dz,droll,dpitch,dyaw,gr_l]
        self.step_move(delta)
        obs = self.get_observation()
        reward = self.get_reward()

        terminated = True if reward == 1 else False
        info = None
        self.steps += 1
        truncated = self.steps >= self.max_steps
        return obs, reward, terminated, truncated, info


    def step_move(self, delta):
        self.robot.move_gripper(delta[-1])
        self.robot.move_ee(delta[:-1]) # delta: dx,dy,dz,droll,dpitch,dyaw,gripper_opening_length
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()


    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)
    









    def get_observation(self):
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
        else:
            assert self.camera is None            
        return depth
    
    def get_reward(self):
        lo, hi = p.getAABB(self.object.id)
        lowest_point_z = lo[2]
        return 1 if lowest_point_z>0.05 else 0







    def reset(self):
        self.robot.reset()
        self.object.reset()
        self.steps = 0
        info = None
        return self.get_observation(), info

    def close(self):
        p.disconnect(self.physicsClient)









    


def make_env():
    obj_pos = (0,0,0)

    cam_pos = (0.2, 0.2, 0.15)
    cam_tar = obj_pos
    cam_up = (0, 0, 1)
    near = 0.1 # 0.01 means anything closer than 1 cm is invisible
    far = 5 # anything further than this is also invisible
    size = (48, 48)
    fov = 40

    rob_pos = (0, 0.5, 0)
    rob_orn = (0, 0, 0)
    ll_t = [-0.25,-0.15,0] # x,y,z
    ul_t = [0.25,0.25,0.25]
    ee_center = np.array([0,0.05,0.25]) # center for starting position of end effector
    ee_tar = np.array(obj_pos) # target position for end effector
    ee_up = np.array([0,-1,0])
    cone_tar = np.array(obj_pos) # target position for the restriction cone
    cone_phi = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c



    object = Object(obj_pos)
    camera = Camera(cam_pos, cam_tar, cam_up, near, far, size, fov)
    robot = Robot(rob_pos, rob_orn, ll_t, ul_t, ee_center, ee_tar, ee_up, cone_tar, cone_phi)



    env = Env(robot, object, camera, vis=True)
    env.reset()
    return env
