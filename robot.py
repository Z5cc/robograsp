import pybullet as p
import math
from collections import namedtuple
from gripper import Gripper
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R



class Robot:

    def __init__(self, pos, ori):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)

    def load(self):
        self.__init_robot__()
        # self.__init_gripper__()

    def __init_robot__(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                                useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        self.eef_id = 8 # link index, not joint index
        
        self.arm_num_dofs = 6
        self.arm_ll = [-3.14159265359,-3,-3.14159265359,-3.14159265359,-3.14159265359,-3.14159265359]
        self.arm_ul = [0,-0.5,3.14159265359,3.14159265359,3.14159265359,3.14159265359]
        self.arm_jr = [u-l for u,l in zip(self.arm_ul,self.arm_ll)]

        self.ll_t = [-0.3,-0.15,0.1] #x,y,z
        self.ul_t = [0.3,0.3,0.3]

        self.c = np.array([0.1,0.1,0.3]) # center for starting position of end effector
        # phi: when taking the z-axis as a vector and rotate it by phi around x, the result is x_c
        # x_c is the center of the restriction cone. x_c is also used as a starting position for the direction of x for the EE
        self.phi = (np.pi/180)*160
        self.alpha_l = (np.pi/180)*35 # alpha_l limits alpha for the restriction cone around x_c
        
        numJoints = p.getNumJoints(self.id)
        self.j_names = []
        self.j_maxForce = []
        self.j_maxVelocity = []
        self.j_dampings = 13*[0.00001]
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            self.j_names.append(info[1].decode("utf-8"))
            self.j_maxForce.append(info[10])
            self.j_maxVelocity.append(info[11])
            jointType = info[2] # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            controllable = (jointType != p.JOINT_FIXED)
            if controllable:
                self.controllable_joints.append(i)
                p.setJointMotorControl2(self.id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]

        
    def __init_gripper__(self):
        self.gripper = Gripper(self.id, self.j_names, self.j_maxForce, self.j_maxVelocity)



    def clamp_t(self,t,ll_t,ul_t):
        t = [max(l, min(x, u)) for x, l, u in zip(t, ll_t, ul_t)]
        return t

    def clamp_r(self,r,phi,alpha_l):
        # caclulate x_c: x_c is the center vector for the restriction cone regarding alpha_l
        x_c = np.array([0,-np.sin(phi),np.cos(phi)])
        # calculate alpha
        x_e = np.array([1,0,0])
        q_e = np.quaternion(0,*x_e) # w,x,y,z
        q_t = r*q_e*r.conj()
        x_t = np.array([q_t.x,q_t.y,q_t.z])
        dot = np.dot(x_c, x_t)
        norm_c = np.linalg.norm(x_c)
        norm_t = np.linalg.norm(x_t)
        alpha = np.arccos(dot / (norm_c * norm_t))

        if alpha>alpha_l:
            # calculate n
            n = np.cross(x_t,x_c)
            n = n/np.linalg.norm(n)
            n_x, n_y, n_z = n[0], n[1], n[2]
            alpha_b = alpha - alpha_l
            sin_half = np.sin(alpha_b/2)
            cos_half = np.cos(alpha_b/2)
            r_back = np.quaternion(cos_half, sin_half*n_x, sin_half*n_y, sin_half*n_z)
            r = r_back*r
            return r
        
        return r

    
    def delta_to_absolute(self, delta):
        dt_EE = np.array(delta[0:3])
        dr_EE = np.array(p.getQuaternionFromEuler(delta[3:6])) # one rotation dr_EE derived from intrinsic euler angles

        # 1. get current end-effector pose in world frame
        state_EE = p.getLinkState(self.id, self.eef_id)
        t = np.array(state_EE[0])  # translation
        r = np.array(state_EE[1])  # quaternion (x,y,z,w)
        
        # 2. translation
        R_W_EE = np.array(p.getMatrixFromQuaternion(r)).reshape(3, 3)
        dt = R_W_EE @ dt_EE
        t += dt

        # 3. rotation
        r = np.quaternion(r[3],r[0],r[1],r[2]) # pybullet quaternion: xyzw  numpy quaternion: wxyz
        dr_EE = np.quaternion(dr_EE[3],dr_EE[0],dr_EE[1],dr_EE[2])
        r = r * dr_EE # do not need to multiplicate with individual like = r*dr_yaw*dr_pitch*dr_roll, because getQuaternionFromEuler is from intrinsic angles
        
        t = t.tolist()

        t = self.clamp_t(t,self.ll_t,self.ul_t)
        r = self.clamp_r(r,self.phi,self.alpha_l)

        r = [r.x,r.y,r.z,r.w]
        return t + r
    




    def move_ee(self, action):
        state_new = self.delta_to_absolute(action)
        pos = state_new[0:3]
        orn = state_new[3:7]
        joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                    jointDamping=self.j_dampings)
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.j_maxForce[joint_id], maxVelocity=self.j_maxVelocity[joint_id])
            
    def move_gripper(self, open_length):
        # self.gripper.move(open_length)
        pass

    def open_gripper(self):
        # self.gripper.open()
        pass

    def close_gripper(self):
        # self.gipper.close()
        pass






    def reset(self):
        self.reset_arm()
        # self.gripper.reset()

    def reset_arm(self):
        """
        reset to rest poses
        """
        x_phi = 0
        y_phi = self.phi - 0.5*np.pi
        z_phi = -0.5*np.pi
        orn = p.getQuaternionFromEuler([x_phi,y_phi,z_phi])
        arm_rest_poses = p.calculateInverseKinematics(self.id, self.eef_id, self.c, orn,
                                                    jointDamping=self.j_dampings)
        for rest_pose, joint_id in zip(arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # Wait for a few steps
        for _ in range(10):
            self.step_simulation()

    def step_simulation(self):
        raise RuntimeError('`step_simulation` method of RobotBase Class should be hooked by the environment.')






    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)
