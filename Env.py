import numpy as np
import pybullet as p
import pybullet_data
import time

from Reward import Reward
from Camera import Camera
from Object import Object
from Robot import Robot


class FailToReachTargetError(RuntimeError):
    pass


class Env:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, robot, object, vis=True) -> None:
        self.vis = vis
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        
        self.action_space_size = 13
        self.steps = 0
        self.max_steps = 100

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.dxin = p.addUserDebugParameter("dx", -0.001, 0.001, 0)
        self.dyin = p.addUserDebugParameter("dy", -0.001, 0.001, 0)
        self.dzin = p.addUserDebugParameter("dz", -0.001, 0.001, 0)
        self.drollId = p.addUserDebugParameter("droll", -0.5, 0.5, 0)
        self.dpitchId = p.addUserDebugParameter("dpitch", -0.5, 0.5, 0)
        self.dyawId = p.addUserDebugParameter("dyaw", -0.5, 0.5, 0)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        
        # LOADING ENTITIES INTO THE ENVIRONMENT
        self.planeID = p.loadURDF("plane.urdf")
        self.robot = robot
        self.robot.load()
        self.object = object
        self.object.load()
        self.camera = Camera(self.robot.id, self.robot.link_map['lens_link'])
        self.reward = Reward(self.robot.id, self.robot.link_map['base_link'], self.robot.link_map['tcp_link'], self.object.id, self.robot.gripper.gripper_range)

        self.reset()


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

        



    def step_demo(self, delta, gr_delta):
        self.robot.move_gripper(gr_delta)
        self.robot.move_tcp(delta, delta_mode=True)
        for _ in range(30):
            self.step_simulation()
        return self.get_observation()


    def step(self, action, gamma):
        print(f'action:{action}')
        if action==0:
            obs = self.grasp()
        else:
            obs = self.seek(action)

        reward = self.get_reward(gamma)
        print(f'reward:{reward}')

        info = None
        self.steps += 1
        truncated = self.steps >= self.max_steps
        terminated = (self.reward.successfull_grasp()==True)  or (self.robot.object_is_in_boundaries(self.object.id)==False)

        return obs, reward, terminated, truncated, info
    

    def grasp(self):
        print('approach')
        self.approach()
        print('close')
        liftable = self.close()
        print('lift')
        self.lift() if liftable else self.retreat()
        return self.get_observation()
    
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
            c=c+1 if self.robot.gripper.has_object(include_delta=True) else 0
            i=i+1
            print(i)
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
            if not self.robot.gripper.has_object(include_delta=False):
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
        elif action==7:
            droll = +dr
        elif action==8:
            droll = -dr
        elif action==9:
            dpitch = +dr
        elif action==10:
            dpitch = -dr
        elif action==11:
            dyaw = +dr
        elif action==12:
            dyaw = -dr
        delta = [dx,dy,dz,droll,dpitch,dyaw]
        # move arm and gripper
        self.robot.move_tcp(delta, delta_mode=True)
        self.robot.open_gripper()
        for _ in range(30):
            self.step_simulation()
        return self.get_observation()



    def step_simulation(self):
        p.stepSimulation()
        if self.vis:
            pass
            # time.sleep(self.SIMULATION_STEP_DELAY)    









    def get_observation(self):
        if isinstance(self.camera, Camera):
            rgb, depth, seg = self.camera.shot()
        else:
            assert self.camera is None
        return depth

    def get_reward(self,gamma):
        return self.reward.get_reward(gamma)




    def reset(self):
        obj_pos = self.object.reset()
        self.robot.reset(obj_pos)
        self.reward.reset()
        for _ in range(30):
            self.step_simulation()

        self.steps = 0
        info = None
        return self.get_observation(), info


    def disconnect(self):
            p.disconnect(self.physicsClient)
