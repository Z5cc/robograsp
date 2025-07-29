import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from Utilities import Object, Camera
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
        obs = self.step_move(delta)
        reward = self.update_reward()
        done = True if reward == 1 else False
        info = 0
        # info = dict(box_opened=self.box_opened, btn_pressed=self.btn_pressed, box_closed=self.box_closed)
        return obs, reward, done, info


    def step_move(self, delta):
        self.robot.move_gripper(delta[-1])
        self.robot.move_ee(delta[:-1]) # delta: dx,dy,dz,droll,dpitch,dyaw,gripper_opening_length
        for _ in range(120):  # Wait for a few steps
            self.step_simulation()
        return self.get_observation()


    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)
    









    def get_observation(self):
        obs = dict()
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
            obs.update(dict(rgb=rgb, depth=depth, seg=seg))
        else:
            assert self.camera is None
            
        obs.update(self.robot.get_joint_obs())
        return obs
    
    def update_reward(self):
        reward = 0
        return reward







    def reset(self):
        self.robot.reset()
        self.object.reset()
        return self.get_observation()

    def close(self):
        p.disconnect(self.physicsClient)
