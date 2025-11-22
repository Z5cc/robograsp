import pybullet as p
import numpy as np
import torch
import time

from CONSTANTS import CONE_CENTER
from assets.robot import Robot
from assets.obj import Obj
from assets.env import Env


def test_action_grasp():
    env = Env()
    env.reset_with_params(tcp_center=CONE_CENTER,tcp_target=(0,0,0),
                          obj_pos=(0,0,0),obj_orn=p.getQuaternionFromEuler((0,0,0)),obj_index=1)
    env.step(0) # action 0 is robot.grasp()
    assert env.reward_handler.successfull_grasp()

def test_ray_offset():
    env = Env()
    env.reset_with_params(tcp_center=np.array([0,0.05,0.20]),tcp_target=(0,0.05,0),
                          obj_pos=(0,0,0),obj_orn=p.getQuaternionFromEuler((0,0,0)),obj_index=1)
    offset = env.reward_handler.ray_offset()
    # print(f'offset: {offset}')
    assert 0.04 < offset < 0.06

def test_ray_reward():
    env = Env()
    env.reset_with_params(tcp_center=np.array([0,0.05,0.20]),tcp_target=(0,0.05,0),
                          obj_pos=(0,0,0),obj_orn=p.getQuaternionFromEuler((0,0,0)),obj_index=1)
    potential_before_step = env.reward_handler.potential
    obs, reward, terminated, truncated, info = env.step(5) # moving in dz local is dy in world
    potential_after_step = env.reward_handler.potential
    # print(f'potential_before_step: {potential_before_step} potential_after_step: {potential_after_step} reward: {reward}')
    assert potential_after_step > potential_before_step
