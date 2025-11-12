import pybullet as p
import numpy as np
import torch

from Robot import Robot
from Obj import Obj
from Env import Env
import train

def test_grasp():
    robot = Robot(TCP_TARGET=(0,0,0))
    obj = Obj(pos=(0,0,0),orn=p.getQuaternionFromEuler((0,0,0)),index=1)
    env = Env(robot,obj,vis=True)
    env.grasp()
    assert env.reward.successfull_grasp()

# def test_update_state():
#     obs = np.array([[2,3],[2,4]])
#     state = torch.tensor([[[1,1],[1,1]],[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]]])
#     new_state = train.update_state(state,obs)
#     assert new_state == torch.tensor([[[2,2],[2,2]],[[3,3],[3,3]],[[4,4],[4,4]],[[2,3],[2,4]]])
