import numpy as np
import pybullet as p
import pybullet_data
import gymnasium as gym
from typing import Optional

from CONSTANTS import *
from assets.rewardhandler import RewardHandler
from assets.actionhandler import ActionHandler
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
        self.reward_handler = RewardHandler(self.robot.id, self.robot.link_map['base_link'], self.robot.link_map['tcp_link'], self.obj.id, self.robot.gripper.gripper_range)
        self.action_handler = ActionHandler()
        # based on load
        self.action_space = gym.spaces.Discrete(N_ACTIONS)
        self.observation_space = gym.spaces.Box(low=self.camera.NEAR, high=self.camera.FAR, shape=(H,W), dtype=np.float32)
        self.reset()


    def step(self, action):
        # print(f'action:{action}')
        if action==0:
            obs = self.grasp()
        else:
            obs = self.seek(action)

        reward = self.get_reward()
        # print(f'reward:{reward}')

        info = {}
        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = (self.reward_handler.successfull_grasp()==True)  or (self.robot.obj_is_in_boundaries(self.obj.id)==False)

        return obs, reward, terminated, truncated, info

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)

        obj_pos = self.obj.reset()
        self.robot.reset(obj_pos)
        self.reward_handler.reset()
        for _ in range(30):
            self.step_simulation()

        self.steps = 0
        obs = self._get_obs()
        info = {}
        return obs, info


    def disconnect(self):
            p.disconnect(self.physicsClient)






    def step_demo(self, delta, gr_delta):
        self.robot.move_gripper(gr_delta)
        self.robot.move_tcp(delta, delta_mode=True)
        for _ in range(30):
            self.step_simulation()
        return self._get_obs()



    def grasp(self):
        # print('approach')
        self.approach()
        # print('close')
        liftable = self.close()
        # print('lift')
        self.lift() if liftable else self.retreat()
        return self._get_obs()
    
    def approach(self):
        dx = 0.005
        delta = [dx,0,0,0,0,0,0]
        x_approach_stop = False
        while not (x_approach_stop):
            x_old = self.robot.get_t_in_tcp_system()[0]
            self.robot.move_tcp(delta,delta_mode=True)
            for _ in range(30):
                self.step_simulation()
            x = self.robot.get_t_in_tcp_system()[0]
            x_approach_stop = x_old+0.9*dx > x # if x does not reach the goal of x_old+0.9*dx

    def close(self):
        liftable=False
        self.robot.close_gripper()
        c,i=0,0
        while (not self.robot.gripper.gr_closed()) and (i<100):
            self.robot.gripper.save_angle()
            for _ in range(60):
                self.step_simulation()
            c=c+1 if self.robot.gripper.has_obj(include_delta=True) else 0
            i=i+1
            # print(i)
            if c==4:
                liftable=True
                break
        return liftable

    def lift(self):
        pos, orn, *_ = p.getLinkState(self.robot.id, self.robot.id_tcp_link)
        pos, orn = list(pos), list(orn)
        while pos[2]<0.2: # lift in z direction
            pos[2]+=0.01
            self.robot.move_tcp(pos+orn)
            for _ in range(30):
                self.step_simulation()
            if not self.robot.gripper.has_obj(include_delta=False):
                self.retreat()
                break

    def retreat(self):
        self.robot.open_gripper()
        delta = [-0.01,0,0,0,0,0,0]
        for _ in range(5):
            self.robot.move_tcp(delta,delta_mode=True)
            for _ in range(30):
                self.step_simulation()


    def seek(self,action):
        # default inits
        dx,dy,dz = 0,0,0
        droll,dpitch,dyaw=0,0,0
        # default deltas
        dt = 0.015
        dr = 0.05
        if action==1:
            dx = +dt
        elif action==2:
            dx = -dt
        elif action==3:
            dy = +dt
        elif action==4:
            dy = -dt
        elif action==5:
            dz = +dt
        elif action==6:
            dz = -dt
        # elif action==7:
        #     droll = +dr
        # elif action==8:
        #     droll = -dr
        # elif action==9:
        #     dpitch = +dr
        # elif action==10:
        #     dpitch = -dr
        # elif action==11:
        #     dyaw = +dr
        # elif action==12:
        #     dyaw = -dr
        delta = [dx,dy,dz,droll,dpitch,dyaw]
        # move arm and gripper
        self.robot.move_tcp(delta, delta_mode=True)
        self.robot.open_gripper()
        for _ in range(30):
            self.step_simulation()
        return self._get_obs()



    def step_simulation(self):
        p.stepSimulation()
        if VIS:
            pass
            # time.sleep(self.SIMULATION_STEP_DELAY)    









    def _get_obs(self):
        return self.camera.shot()

    def get_reward(self):
        return self.reward_handler.get_reward()





