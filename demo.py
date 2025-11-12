import pybullet as p
import time

from assets.robot import Robot
from assets.obj import Obj
from assets.env import Env


def read_debug_parameter(dx_in, dy_in, dz_in, droll_in, dpitch_in, dyaw_in, gr_in):
    # read the value of task parameter
    dx = p.readUserDebugParameter(dx_in)
    dy = p.readUserDebugParameter(dy_in)
    dz = p.readUserDebugParameter(dz_in)
    droll = p.readUserDebugParameter(droll_in)
    dpitch = p.readUserDebugParameter(dpitch_in)
    dyaw = p.readUserDebugParameter(dyaw_in)
    gr = p.readUserDebugParameter(gr_in)
    return dx, dy, dz, droll, dpitch, dyaw, gr


env = Env(Robot(),Obj())
# custom sliders to tune parameters (name of the parameter,range,initial value)
dx_in = p.addUserDebugParameter("dx", -0.001, 0.001, 0)
dy_in = p.addUserDebugParameter("dy", -0.001, 0.001, 0)
dz_in = p.addUserDebugParameter("dz", -0.001, 0.001, 0)
droll_in = p.addUserDebugParameter("droll", -0.5, 0.5, 0)
dpitch_in = p.addUserDebugParameter("dpitch", -0.5, 0.5, 0)
dyaw_in = p.addUserDebugParameter("dyaw", -0.5, 0.5, 0)
gr_in = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

while True:
    debug_parameter = read_debug_parameter(dx_in, dy_in, dz_in, droll_in, dpitch_in, dyaw_in, gr_in)
    delta, gr_delta = debug_parameter[0:6], debug_parameter[-1]
    obs = env.step_demo(delta,gr_delta)
    # lo, hi = p.getAABB(obj.id)
    # print(f'lo:{lo}')
    # print(f'hi:{hi}')
    # print(f'obs:{obs}')
    # print(f'shoulder_torque:{env.robot.get_shoulder_torque()}')
    # print(f'gripper_torque:{env.robot.gripper.get_torque()}')
    # print(f'ray_offest:{env.reward.ray_offset()}')
