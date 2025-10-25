import pybullet as p

from Robot import Robot
from Object import Object
from Env import Env


def test_grasp():
    robot = Robot(TCP_TARGET=(0,0,0))
    object = Object(pos=(0,0,0),orn=p.getQuaternionFromEuler((0,0,0)),index=1)
    env = Env(robot,object,vis=True)
    env.grasp()
    assert env.reward.successfull_grasp()
