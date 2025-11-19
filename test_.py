import pybullet as p
import numpy as np
import torch

from assets.robot import Robot
from assets.obj import Obj
from assets.env import Env


def test_action_grasp():
    robot = Robot(tcp_target=(0,0,0))
    obj = Obj(pos=(0,0,0),orn=p.getQuaternionFromEuler((0,0,0)),index=1)
    env = Env(robot,obj)
    env.step(0)
    assert env.reward_handler.successfull_grasp()

def test_robot():
    pass
# def test_update_state():
#     obs = np.array([[2,3],[2,4]])
#     state = torch.tensor([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]]])
#     new_state = train.update_state(state,obs)
#     assert new_state == torch.tensor([[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]],[[2,3],[2,4]]])
