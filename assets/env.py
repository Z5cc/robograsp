import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
import time
import random
from typing import Optional

from CONSTANTS import VIS, REALTIME, SIMULATION_STEP_DELAY, N_ACTIONS, H, W, CONE_CENTER
from assets.util import random_quaternion
from assets.rewardhandler import RewardHandler
from assets.camera import Camera
from assets.robot import Robot
from assets.obj import Obj


class Env(gym.Env):

    def __init__(self) -> None:
        self.physicsClient = p.connect(p.GUI if VIS else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.steps = 0
        self.max_steps = 100
        # load assets
        self.planeID = p.loadURDF("plane.urdf")
        self.robot = Robot()
        self.obj = Obj()
        self.camera = Camera(self.robot)
        self.reward_handler = RewardHandler(self.robot, self.obj)
        # based on load
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=self.camera.NEAR, high=self.camera.FAR, shape=(H,W), dtype=np.float32)


    def step(self, action):
        if action==0:
            steps_generator = self.robot.grasp()
        elif 0<action<N_ACTIONS:    # N_ACTIONS=13 leads to action=0 until action=12
            steps_generator = self.robot.seek(action)
        else:
            steps_generator = [30]
        
        for steps in steps_generator:
            for  s in range(steps):
                self._step_simulation()

        obs = self._get_obs()
        reward = self._get_reward()
        
        info = {}
        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = (self.reward_handler.successfull_grasp()==True)  or (self.obj.is_in_boundaries()==False)
        return obs, reward, terminated, truncated, info
    
    def step_user_control(self, delta, gr_delta):
        self.robot.move_gripper(gr_delta)
        self.robot.move_tcp_delta(delta)
        for _ in range(30):
            self._step_simulation()
        return self._get_obs()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        dev = 0.04
        obj_pos = [random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),0]
        obj_orn = random_quaternion()
        obj_index = random.randint(0,999)
        tcp_center = CONE_CENTER+np.array([random.uniform(-dev,dev),random.uniform(-dev,dev),random.uniform(-dev,dev)])
        tcp_target = obj_pos+np.array([random.uniform(-dev,dev),random.uniform(-dev,dev),0])
        
        obs, info = self.reset_with_params(obj_pos, obj_orn, obj_index, tcp_center, tcp_target)
        return obs, info

    def close(self):
            p.disconnect(self.physicsClient)


    def reset_with_params(self, obj_pos, obj_orn, obj_index, tcp_center, tcp_target):
        
        self.obj.reset(obj_pos, obj_orn, obj_index)
        self.robot.reset(tcp_center, tcp_target)
        self.reward_handler.reset()
        for _ in range(30):
            self._step_simulation()
        obs = self._get_obs()
        self.steps = 0
        info = {}
        return obs, info

    def _step_simulation(self):
        p.stepSimulation()
        if VIS and REALTIME:
            time.sleep(SIMULATION_STEP_DELAY)    

    def _get_obs(self):
        return self.camera.shot()

    def _get_reward(self):
        return self.reward_handler.get_reward()
