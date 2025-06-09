import pybullet as p
import math
from collections import namedtuple
from gripper import Gripper
import numpy as np
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
        self.arm_rest_poses = [-1.5690622952052096, -1.5446774605904932, 1.343946009733127, -1.3708613585093699,
                                -1.5707970583733368, 0.0009377758247187636]
        
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



    
    def delta_to_absolute(self, delta):
        dx, dy, dz, droll, dpitch, dyaw = delta

        # 1. get current end-effector pose in world frame
        state_ee = p.getLinkState(self.id, self.eef_id)
        t_ee = np.array(state_ee[0])  # translation
        rot_ee = np.array(state_ee[1])  # quaternion (x,y,z,w)

        # 2. translation
        R_ee = np.array(p.getMatrixFromQuaternion(rot_ee)).reshape(3, 3)
        dt_ee = np.array([dx, dy, dz])
        dt_world = R_ee @ dt_ee
        t_new = t_ee + dt_world

        # 3. rotation
        


        # 3. build delta transformation in local frame
        
        delta_rot = R.from_euler('xyz', [droll, dpitch, dyaw]).as_matrix()

        T_delta = np.eye(4)
        T_delta[:3, :3] = delta_rot
        T_delta[:3, 3] = delta_trans

        # 4. compute new pose in world frame
        T_target = T_current @ T_delta

        new_pos = T_target[:3, 3]
        new_rot = R.from_matrix(T_target[:3, :3]).as_euler('xyz')

        return t_new.tolist() + new_rot.tolist()
    




    def move_ee(self, action, control_method):
        assert control_method in ('joint', 'end')
        if control_method == 'end':
            x, y, z, roll, pitch, yaw = self.delta_to_absolute(action)
            pos = (x, y, z)
            orn = p.getQuaternionFromEuler((roll, pitch, yaw))
            joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn,
                                                       self.arm_ll, self.arm_ul, self.arm_jr, self.arm_rest_poses,
                                                      maxNumIterations=20,jointDamping=self.j_dampings)
        elif control_method == 'joint':
            assert len(action) == self.arm_num_dofs
            joint_poses = action
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
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
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
