import pybullet as p
import numpy as np
import quaternion
from scipy.spatial.transform import Rotation as R

from Gripper import Gripper




class Joint():
    def __init__(self, index, joint_type, max_force, max_vel, controllable, numeric_damping=0.00001):
        self.index = index
        self.joint_type = joint_type
        self.max_force = max_force
        self.max_vel = max_vel
        self.controllable = controllable
        self.numeric_damping = numeric_damping



class Robot:

    def __init__(self, pos, ori, ll_t, ul_t, tcp_center, tcp_tar, tcp_up, cone_tar, cone_phi, object):
        self.base_pos = pos
        self.base_ori = p.getQuaternionFromEuler(ori)
        
        self.ll_t = ll_t
        self.ul_t = ul_t
        self.tcp_center = tcp_center
        self.tcp_tar = tcp_tar
        self.tcp_up = tcp_up
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

        self.link_map = {} # {link_name: index}
        self.joint_map = {} # {joint_name: index}

        self.joints = {} # {index: Joint}
        self.joints_dampings = []

        self.joints_controllable_ids = [] # [indeces]
        self.joints_controllable_arm_ids = [] # [indeces]
        
        numJoints = p.getNumJoints(self.id)
        for i in range(numJoints):
            info = p.getJointInfo(self.id, i)

            joint_name = info[1].decode("utf-8")
            link_name = info[12].decode("utf-8")

            index = info[0]
            joint_type = info[2] # JOINT_REVOLUTE, JOINT_PRISMATIC, JOINT_SPHERICAL, JOINT_PLANAR, JOINT_FIXED
            max_force = info[10]
            max_vel = info[11]
            controllable = (joint_type != p.JOINT_FIXED)
            joint = Joint(index,joint_type,max_force,max_vel,controllable)
            self.joints_dampings.append(joint.numeric_damping)

            self.link_map[link_name] = index
            self.joint_map[joint_name] = index
            self.joints[index] = joint

            if controllable:
                self.joints_controllable_ids.append(index)
                if len(self.joints_controllable_ids)<=self.arm_num_dofs:
                    self.joints_controllable_arm_ids.append(index)

        for joint_id in self.joints_controllable_ids:
            p.setJointMotorControl2(self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        
        self.id_tcp_link = self.link_map['tcp_link']
        self.gripper = Gripper(self.id, self.link_map ,self.joint_map, self.joints, self.object)
        

    




    def move_tcp(self, delta):
        state_new = self.delta_to_absolute(delta)
        pos = state_new[0:3]
        orn = state_new[3:7]
        joint_poses = p.calculateInverseKinematics(self.id, self.id_tcp_link, pos, orn, jointDamping=self.joints_dampings)
        # arm
        for joint_pose, joint_id in zip(joint_poses, self.joints_controllable_arm_ids):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_pose,
                                    force=self.joints[joint_id].max_force, maxVelocity=self.joints[joint_id].max_vel)
            
    def delta_to_absolute(self, delta):
        dt_TCP = np.array(delta[0:3])
        dr_TCP = np.array(p.getQuaternionFromEuler(delta[3:6])) # one rotation dr_TCP derived from intrinsic euler angles

        # 1. get current tcp pose in world frame
        state_TCP = p.getLinkState(self.id, self.id_tcp_link)
        t = np.array(state_TCP[0])  # translation
        r = np.array(state_TCP[1])  # quaternion (x,y,z,w)
        
        # 2. translation
        R_W_TCP = np.array(p.getMatrixFromQuaternion(r)).reshape(3, 3)
        dt = R_W_TCP @ dt_TCP
        t += dt

        # 3. rotation
        r = np.quaternion(r[3],r[0],r[1],r[2]) # pybullet quaternion: xyzw  numpy quaternion: wxyz
        dr_TCP = np.quaternion(dr_TCP[3],dr_TCP[0],dr_TCP[1],dr_TCP[2])
        r = r * dr_TCP # do not need to multiplicate with individual like = r*dr_yaw*dr_pitch*dr_roll, because getQuaternionFromEuler is from intrinsic angles
        
        t = t.tolist()

        t = self.clamp_t(t,self.ll_t,self.ul_t)
        r = self.clamp_r(r,self.tcp_center,self.cone_tar,self.cone_phi)

        r = [r.x,r.y,r.z,r.w]
        return t + r

    def clamp_t(self,t,ll_t,ul_t):
        t = [max(l, min(x, u)) for x, l, u in zip(t, ll_t, ul_t)]
        return t

    def clamp_r(self,r,tcp_center,cone_tar,cone_phi):
        # caclulate cone_vec: cone_vec is the center vector for the restriction cone regarding cone_phi
        cone_vec = cone_tar - tcp_center
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
        tcp_vec = self.tcp_tar - self.tcp_center
        tcp_vec = tcp_vec / np.linalg.norm(tcp_vec)
        z_new = self.tcp_up - np.dot(self.tcp_up,tcp_vec)*tcp_vec  # z_new = up - proj. of up on tcp_vec
        z_new = z_new / np.linalg.norm(z_new)
        y_new = np.cross(z_new,tcp_vec)
        y_new = y_new / np.linalg.norm(y_new)
        R_new = np.column_stack((tcp_vec,y_new,z_new))
        R_new = R.from_matrix(R_new)

        orn = R_new.as_quat().tolist()

        # 1. deactivate motors first to avoid driving back to old position after p.resetJointState
        for joint_id in self.joints_controllable_arm_ids:
            p.setJointMotorControl2(self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        # 2. p.resetJointState to precalculated arm positions.
        # this intermediate fixed positions are needed, so that in next step no undesired positions are returned by InverseKinematics
        for rest_pose, joint_id in zip(self.arm_rest_poses, self.joints_controllable_arm_ids):
            p.resetJointState(self.id, joint_id, rest_pose)

        # 3. p.resetJointState to new calculated arm positions
        arm_rest_poses = p.calculateInverseKinematics(self.id, self.id_tcp_link, self.tcp_center, orn, jointDamping=self.joints_dampings)
        for rest_pose, joint_id in zip(arm_rest_poses, self.joints_controllable_arm_ids):
            p.resetJointState(self.id, joint_id, rest_pose)

        # 4. drive motors to reseted joint states to hold new position
        for rest_pose, joint_id in zip(arm_rest_poses, self.joints_controllable_arm_ids):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, rest_pose,
                            force=self.joints[joint_id].max_force, maxVelocity=self.joints[joint_id].max_vel)

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
        tcp_pos = p.getLinkState(self.id, self.id_tcp_link)[0]
        return dict(positions=positions, velocities=velocities, tcp_pos=tcp_pos)
