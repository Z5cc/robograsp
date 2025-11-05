import pybullet as p
import time

from Robot import Robot
from Object import Object
from Env import Env


def user_control_demo():
    robot = Robot()
    object = Object()
    env = Env(robot, object)
    while True:
        debug_parameter = env.read_debug_parameter()
        delta, gr_delta = debug_parameter[0:6], debug_parameter[-1]

        obs = env.step_demo(delta,gr_delta)

        # lo, hi = p.getAABB(object.id)
        # print(f'lo:{lo}')
        # print(f'hi:{hi}')
        # print(f'obs:{obs}')
        # print(f'shoulder_torque:{env.robot.get_shoulder_torque()}')
        # print(f'gripper_torque:{env.robot.gripper.get_torque()}')
        # print(f'ray_offest:{env.reward.ray_offset()}')




user_control_demo()
