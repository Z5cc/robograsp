import pybullet as p
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R

from Gripper import Gripper



class Robot:

    def __init__(self, pos, ori, ll_t, ul_t, ee_center, ee_tar, ee_up, cone_tar, cone_phi, object):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        
        self.ll_t = ll_t
        self.ul_t = ul_t
        self.ee_center = ee_center
        self.ee_tar = ee_tar
        self.ee_up = ee_up
        self.cone_tar = cone_tar
        self.cone_phi = cone_phi
        self.object = object

    def load(self):
        self.id = p.loadURDF('./urdf/ur5_robotiq_85.urdf', self.base_pos, self.base_ori,
                                useFixedBase=True, flags=p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        
        self.arm_num_dofs = 6
        self.arm_rest_poses = [-1.8427108144422384,-1.783986598255091,1.9232743283452045,-1.9004039537122694,-1.5180998101236258,-0.2668835598602039]
        self.arm_ll = [-3.14159265359,-3,-3.14159265359,-3.14159265359,-3.14159265359,-3.14159265359]
        self.arm_ul = [0,-0.5,3.14159265359,3.14159265359,3.14159265359,3.14159265359]
        self.arm_jr = [u-l for u,l in zip(self.arm_ul,self.arm_ll)]

        numJoints = p.getNumJoints(self.id)
        self.j_names = []
        self.j_maxForce = []
        self.j_maxVelocity = []
        self.controllable_joints = []
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)
            name = info[1].decode("utf-8")
            jointType = info[2] # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            controllable = (jointType != p.JOINT_FIXED)
            self.j_names.append(name)
            self.j_maxForce.append(info[10])
            self.j_maxVelocity.append(info[11])
            if controllable:
                self.controllable_joints.append(i)
                p.setJointMotorControl2(self.id, i, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
            if name == 'ee_tcp_joint':
                self.eef_id = i # link index, not joint index. however the joint index i will have same value as link index
            if name == 'robotiq_85_base_joint':
                gripper_base_link_id = i
        assert len(self.controllable_joints) >= self.arm_num_dofs
        self.arm_controllable_joints = self.controllable_joints[:self.arm_num_dofs]
        
        self.gripper = Gripper(self.id, gripper_base_link_id, self.j_names, self.j_maxForce, self.j_maxVelocity, self.object)
        

    




    def move_ee(self, delta):
        state_new = self.delta_to_absolute(delta)
        pos = state_new[0:3]
        orn = state_new[3:7]
        joint_poses = p.calculateInverseKinematics(self.id, self.eef_id, pos, orn)
        # arm
        for i, joint_id in enumerate(self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_poses[i],
                                    force=self.j_maxForce[joint_id], maxVelocity=self.j_maxVelocity[joint_id])
            
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
        r = self.clamp_r(r,self.ee_center,self.cone_tar,self.cone_phi)

        r = [r.x,r.y,r.z,r.w]
        return t + r

    def clamp_t(self,t,ll_t,ul_t):
        t = [max(l, min(x, u)) for x, l, u in zip(t, ll_t, ul_t)]
        return t

    def clamp_r(self,r,ee_center,cone_tar,cone_phi):
        # caclulate cone_vec: cone_vec is the center vector for the restriction cone regarding cone_phi
        cone_vec = cone_tar - ee_center
        cone_vec = cone_vec / np.linalg.norm(cone_vec)
        # calculate alpha
        x_e = np.array([1,0,0])
        q_e = np.quaternion(0,*x_e) # w,x,y,z
        q_t = r*q_e*r.conj()
        x_t = np.array([q_t.x,q_t.y,q_t.z])
        x_t = x_t / np.linalg.norm(x_t)
        alpha = np.arccos(np.dot(cone_vec, x_t))

        if alpha>cone_phi:
            # calculate n
            n = np.cross(x_t,cone_vec)
            n = n/np.linalg.norm(n)
            n_x, n_y, n_z = n[0], n[1], n[2]
            alpha_b = alpha - cone_phi
            sin_half = np.sin(alpha_b/2)
            cos_half = np.cos(alpha_b/2)
            r_back = np.quaternion(cos_half, sin_half*n_x, sin_half*n_y, sin_half*n_z)
            r = r_back*r
            return r
        
        return r

    
    




    def move_gripper(self, open_length):
        self.gripper.move(open_length)

    def open_gripper(self):
        self.gripper.open()

    def close_gripper(self):
        self.gripper.close()










    def reset(self):
        """
        reset to rest poses
        """
        ee_vec = self.ee_tar - self.ee_center
        ee_vec = ee_vec / np.linalg.norm(ee_vec)
        z_new = self.ee_up - np.dot(self.ee_up,ee_vec)*ee_vec  # z_new = up - proj. of up on ee_vec
        z_new = z_new / np.linalg.norm(z_new)
        y_new = np.cross(z_new,ee_vec)
        y_new = y_new / np.linalg.norm(y_new)
        R_new = np.column_stack((ee_vec,y_new,z_new))
        R_new = R.from_matrix(R_new)

        orn = R_new.as_quat().tolist()

        # 1. deactivate motors first to avoid driving back to old position after p.resetJointState
        for joint_id in self.arm_controllable_joints:
            p.setJointMotorControl2(self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # 2. p.resetJointState to precalculated arm positions.
        # this intermediate fixed positions are needed, so that in next step no undesired positions are returned by InverseKinematics
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # 3. p.resetJointState to new calculated arm positions
        arm_rest_poses = p.calculateInverseKinematics(self.id, self.eef_id, self.ee_center, orn)
        for rest_pose, joint_id in zip(arm_rest_poses, self.arm_controllable_joints):
            p.resetJointState(self.id, joint_id, rest_pose)

        # 4. drive motors to reseted joint states to hold new position
        for rest_pose, joint_id in zip(arm_rest_poses, self.arm_controllable_joints):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, rest_pose,
                            force=self.j_maxForce[joint_id], maxVelocity=self.j_maxVelocity[joint_id])

        # drive gripper to default open position
        self.gripper.reset()

        # 5. let world settle -> done in env, because besides robot object also needs to settle


    def get_joint_obs(self):
        positions = []
        velocities = []
        for joint_id in self.controllable_joints:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        ee_pos = p.getLinkState(self.id, self.eef_id)[0]
        return dict(positions=positions, velocities=velocities, ee_pos=ee_pos)
