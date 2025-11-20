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

def test_reward():
    env = Env()
    env.reset_with_params(tcp_center=CONE_CENTER,tcp_target=(0,0,0),
                          obj_pos=(0,0,0),obj_orn=p.getQuaternionFromEuler((0,0,0)),obj_index=1)
    offset = env.reward_handler.ray_offset()
    print(f'offset::::::::{offset}')
    time.sleep(10)
    assert offset > 0.01

# def test_update_state():
#     obs = np.array([[2,3],[2,4]])
#     state = torch.tensor([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]]])
#     new_state = train.update_state(state,obs)
#     assert new_state == torch.tensor([[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]],[[2,3],[2,4]]])
