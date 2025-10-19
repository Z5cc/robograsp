import pybullet as p

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

        lo, hi = p.getAABB(object.id)
        print('\n\n\nlo: ',lo)
        print('\n\n\nhi: ',hi)
        # print('\n\n\n','obsbeginn',obs,'obsend')
        # print('\n\n\n','shoulder_torque: ',env.robot.get_shoulder_torque())
        print('\n\n\n','gripper_torque: ',env.robot.gripper.get_torque())









user_control_demo()
