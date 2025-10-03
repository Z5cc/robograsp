import numpy as np
import pybullet as p
import pybullet_data

from Reward import Reward
from Camera import Camera
from Object import Object
from Robot import Robot
from tqdm import tqdm


class FailToReachTargetError(RuntimeError):
    pass


class Env:

    SIMULATION_STEP_DELAY = 1 / 240.

    def __init__(self, vis=True) -> None:
        self.vis = vis
        if self.vis:
            self.p_bar = tqdm(ncols=0, disable=False)
        self.physicsClient = p.connect(p.GUI if self.vis else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)

        self.action_space_size = 13
        self.steps = 0
        self.max_steps = 100

        # custom sliders to tune parameters (name of the parameter,range,initial value)
        self.dxin = p.addUserDebugParameter("dx", -0.1, 0.1, 0)
        self.dyin = p.addUserDebugParameter("dy", -0.1, 0.1, 0)
        self.dzin = p.addUserDebugParameter("dz", -0.1, 0.1, 0)
        self.drollId = p.addUserDebugParameter("droll", -0.5, 0.5, 0)
        self.dpitchId = p.addUserDebugParameter("dpitch", -0.5, 0.5, 0)
        self.dyawId = p.addUserDebugParameter("dyaw", -0.5, 0.5, 0)
        self.gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length", 0, 0.085, 0.04)

        
        # LOADING
        near = 0.001 # 0.1 means anything closer than 10 cm is invisible
        far = 0.6 # anything further than this is also default fovdefault fov invisible
        size = (16, 16)
        fov = 50

        rob_pos = (0, 0.5, 0)
        rob_orn = (0, 0, 0)
        ll_t = [-0.15,-0.15,0.03] # x,y,z
        ul_t = [0.15,0.15,0.20]
        tcp_center = np.array([0,0.05,0.20]) # center for starting position of tcp
        tcp_up = np.array([0,-1,0])
        cone_tar = np.array([0,0,0]) # target position for the restriction cone
        cone_phi = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c

        self.planeID = p.loadURDF("plane.urdf")
        self.object = Object(ll_t,ul_t)
        self.robot = Robot(rob_pos, rob_orn, ll_t, ul_t, tcp_center, tcp_up, cone_tar, cone_phi)
        self.camera = Camera(self.robot.id, self.robot.link_map['lens_link'],  near, far, size, fov)
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

        






    def step(self, action, gamma):
        gr_delta = 'open'
        dx,dy,dz = 0,0,0
        droll,dpitch,dyaw=0,0,0
        dt = 0.015
        dr = 0.1

        if action==0:
            gr_delta = 'close'
        elif action==1:
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
        obs, graspable = self.step_move(delta,gr_delta)
        reward = self.get_reward(gr_delta,gamma)
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

    def get_reward(self,gr_delta,gamma):
        return self.reward.get_reward(gr_delta,gamma)




    def reset(self):
        obj_pos = self.object.reset()
        self.robot.reset(obj_pos)
        self.reward.reset()
        for _ in range(30):
            self.step_simulation()

        self.steps = 0
        info = None
        return self.get_observation(), info

    def close(self):
        p.disconnect(self.physicsClient)
