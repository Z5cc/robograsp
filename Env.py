import time
import math
import random

import numpy as np
import pybullet as p
import pybullet_data

from Utilities import Object, Camera
from Robot import Robot
from collections import namedtuple
from attrdict import AttrDict
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class Env:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, object, camera=None, vis=False) -> None:
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.camera = camera
        self.object = object
        self.robot = robot

        # load
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeID = p.loadURDF("plane.urdf")
        self.object.load()
        self.robot.load()
        self.camera.load(robot)

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.dxin = p.addUserDebugParameter("dx", -0.1, 0.1, 0)
        self.dyin = p.addUserDebugParameter("dy", -0.1, 0.1, 0)
        self.dzin = p.addUserDebugParameter("dz", -0.1, 0.1, 0)
        self.drollId = p.addUserDebugParameter("droll", -0.5, 0.5, 0)
        self.dpitchId = p.addUserDebugParameter("dpitch", -0.5, 0.5, 0)
        self.dyawId = p.addUserDebugParameter("dyaw", -0.5, 0.5, 0)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        self.action_space_size = 7
        self.steps = 0
        self.max_steps = 100

    def read_debug_parameter(self):
        # read the value of task parameter
        dx = p.readUserDebugParameter(self.dxin)
        dy = p.readUserDebugParameter(self.dyin)
        dz = p.readUserDebugParameter(self.dzin)
        droll = p.readUserDebugParameter(self.drollId)
        dpitch = p.readUserDebugParameter(self.dpitchId)
        dyaw = p.readUserDebugParameter(self.dyawId)
        gripper_opening_length = p.readUserDebugParameter(self.gripper_opening_length_control)

        return dx, dy, dz, droll, dpitch, dyaw, gripper_opening_length

        






    def step(self, action):
        gr_delta = 'open'
        dx,dy,dz = 0,0,0
        droll,dpitch,dyaw=0,0,0

        if action==0: # grasp
            gr_delta = 'close'
        elif action==1: # approach
            dx = 0.015
        elif action==2: # regrasp
            dx = -0.015
        elif action==3: # adjust while moving little to object
            dy = 0.015
        elif action==4:
            dy = -0.015
        elif action==5:
            dz = +0.015
        elif action==6:
            dz = -0.015

        delta = [dx,dy,dz,droll,dpitch,dyaw]
        obs, graspable = self.step_move(delta,gr_delta)
        reward = self.get_reward(gr_delta)
        print(f'reward:{reward}\n')

        info = None
        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = (graspable==True)  or (self.object.is_in_boundaries()==False)

        return obs, reward, terminated, truncated, info


    def step_move(self, delta, gr_delta):
        
        # move gripper
        graspable=False
        if gr_delta=='close':
            self.robot.close_gripper()
            c=0
            while not self.robot.gripper.gr_closed():
                self.robot.gripper.save_angle()
                for _ in range(60):
                    self.step_simulation()
                c=c+1 if self.robot.gripper.has_object() else 0

                if c==4:
                    graspable=True
                    delta[0]=-0.5
                    break

        elif gr_delta=='open':
            self.robot.open_gripper()
        else:
            self.robot.move_gripper(gr_delta)
        
        # move arm
        self.robot.move_tcp(delta)
        for _ in range(30):
            self.step_simulation()
        
        return self.get_observation(), graspable


    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            # time.sleep(self.SIMULATION_STEP_DELAY)
            self.p_bar.update(1)
    









    def get_observation(self):
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
        else:
            assert self.camera is None            
        return depth
    
    def distance_function(self,x,rew_at_one=-50):
        return rew_at_one*x

    def get_reward(self,gr_delta):

        lo, hi = p.getAABB(self.object.id)
        lowest_point_z = lo[2]
        # 1: successfull grasp reward
        if lowest_point_z>0.05:
            return 100
        

        else:
            r_cap = -10
            r_op = -5 if gr_delta=='close' else 0
            r_gr = 0

            object_hit,d,delta,graspable = self.robot.gripper.ray_tests()
            if object_hit:
                r_d = self.distance_function(d)

                # 2: object ready to be grasped [necessity: r<1]
                if graspable:
                    r_gr = 10
                # 3: object in front of gripper [necessity: r<1]
                elif delta>0.060:
                    r_gr = 5
                elif 0<delta<0.060:
                    r_gr = 1+(4/0.060)*delta
                else:
                    r_gr = 1
            else:
                # 4: object not in front of gripper [necessity: r<1]
                offset = self.robot.gripper.ray_offset()
                r_d = self.distance_function(offset)

            return r_d + r_op + r_cap + r_gr






    def reset(self):
        self.robot.reset()
        self.object.reset()
        for _ in range(30):
            self.step_simulation()

        self.steps = 0
        info = None
        return self.get_observation(), info

    def close(self):
        p.disconnect(self.physicsClient)









    


def make_env():
    obj_pos = (0,0,0)

    near = 0.001 # 0.1 means anything closer than 10 cm is invisible
    far = 0.6 # anything further than this is also default fovdefault fov invisible
    size = (16, 16)
    fov = 50

    rob_pos = (0, 0.5, 0)
    rob_orn = (0, 0, 0)
    ll_t = [-0.15,-0.15,0.03] # x,y,z
    ul_t = [0.15,0.15,0.25]
    tcp_center = np.array([0,0.05,0.25]) # center for starting position of tcp
    tcp_tar = np.array(obj_pos) # target position for tcp
    tcp_up = np.array([0,-1,0])
    cone_tar = np.array(obj_pos) # target position for the restriction cone
    cone_phi = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c



    object = Object(obj_pos, ll_t, ul_t)
    robot = Robot(rob_pos, rob_orn, ll_t, ul_t, tcp_center, tcp_tar, tcp_up, cone_tar, cone_phi,object)
    camera = Camera(near, far, size, fov)



    env = Env(robot, object, camera, vis=True)
    env.reset()
    return env
