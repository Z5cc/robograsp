import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
import time
from typing import Optional

from CONSTANTS import *
from assets.rewardhandler import RewardHandler
from assets.camera import Camera


class Env(gym.Env):

    def __init__(self, robot, obj) -> None:
        self.physicsClient = p.connect(p.GUI if VIS else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.steps = 0
        self.max_steps = 100
        # load physical assets
        self.planeID = p.loadURDF("plane.urdf")
        self.robot = robot
        self.robot.load()
        self.obj = obj
        self.obj.load()
        self.camera = Camera(self.robot.id, self.robot.link_map['lens_link'])
        # load handler
        self.reward_handler = RewardHandler(self.robot.id, self.robot.link_map['base_link'], self.robot.link_map['tcp_link'], self.obj.id, self.robot.get_gripper_range())
        # based on load
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=self.camera.NEAR, high=self.camera.FAR, shape=(H,W), dtype=np.float32)
        self.reset()


    def step(self, action):
        if action==0:
            steps_generator = self.robot.grasp()
        elif 0<action<N_ACTIONS:    # N_ACTIONS=13 leads to action=0 until action=12
            steps_generator = self.robot.seek(action)
        else:
            raise ValueError
        
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
            self.step_simulation()
        return self._get_obs()

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        obj_pos = self.obj.reset()
        self.robot.reset(obj_pos)
        self.reward_handler.reset()
        for _ in range(30):
            self._step_simulation()

        obs = self._get_obs()
        self.steps = 0
        info = {}
        return obs, info

    def close(self):
            p.disconnect(self.physicsClient)


    def _step_simulation(self):
        p.stepSimulation()
        if VIS:
            pass
            # time.sleep(SIMULATION_STEP_DELAY)    

    def _get_obs(self):
        return self.camera.shot()

    def _get_reward(self):
        return self.reward_handler.get_reward()
