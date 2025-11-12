import pybullet as p
import numpy as np
import quaternion
import random
from scipy.spatial.transform import Rotation as R

from assets.gripper import Gripper



class Joint():
    def __init__(self, index, joint_type, max_force, max_vel, controllable):
        self.index = index
        self.joint_type = joint_type
        self.max_force = max_force
        self.max_vel = max_vel
        self.controllable = controllable
        self.NUMERIC_DAMPING = 0.00001



class Robot:

    def __init__(self, TCP_TARGET=None):
        self.random = TCP_TARGET is None
        self.BASE_POS = (0,0.5,0)
        self.BASE_ORN = p.getQuaternionFromEuler((0,0,0))
        self.LL_T = [-0.15,-0.15,0.03] # x,y,z
        self.UL_T = [0.15,0.15,0.20]
        self.TCP_CENTER = np.array([0,0.05,0.20]) # center for starting position of tcp
        self.TCP_TARGET = TCP_TARGET
        self.TCP_UP = np.array([0,-1,0])
        self.CONE_TAR = np.array([0,0,0]) # target position for the restriction cone
        self.CONE_PHI = (np.pi/180)*35 # cone_phi limits alpha for the restriction cone around x_c


    def load(self):
        # LOADING
        self.id = p.loadURDF('./assets/urdf/ur5_robotiq_85.urdf', self.BASE_POS, self.BASE_ORN,
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
            self.joints_dampings.append(joint.NUMERIC_DAMPING)

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
        self.gripper = Gripper(self.id, self.link_map ,self.joint_map, self.joints)

    def move_tcp(self, target, delta_mode=False):
        if delta_mode:
            target = self.delta_to_absolute(target)
        pos = target[0:3]
        orn = target[3:7]
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
        R_TCP_W = np.array(p.getMatrixFromQuaternion(r)).reshape(3, 3)
        dt = R_TCP_W @ dt_TCP # from local to global world frame
        t += dt

        # 3. rotation
        r = np.quaternion(r[3],r[0],r[1],r[2]) # pybullet quaternion: xyzw  numpy quaternion: wxyz
        dr_TCP = np.quaternion(dr_TCP[3],dr_TCP[0],dr_TCP[1],dr_TCP[2])
        r = r * dr_TCP # do not need to multiplicate with individual like = r*dr_yaw*dr_pitch*dr_roll, because getQuaternionFromEuler is from intrinsic angles
        
        t = t.tolist()

        t = self.clamp_t(t,self.LL_T,self.UL_T)
        r = self.clamp_r(r,self.TCP_CENTER,self.CONE_TAR,self.CONE_PHI)

        r = [r.x,r.y,r.z,r.w]
        return t + r
    
    def get_t_in_tcp_system(self):
        """
        return t, but in the tcp coordinate system with the axis according to current tcp orientation
        """
        state_TCP = p.getLinkState(self.id, self.id_tcp_link)
        t = np.array(state_TCP[0])  # translation
        r = np.array(state_TCP[1])  # quaternion (x,y,z,w)

        R_TCP_W = np.array(p.getMatrixFromQuaternion(r)).reshape(3, 3)
        R_W_TCP = np.transpose(R_TCP_W)
        t = R_W_TCP @ t
        return t

    def clamp_t(self,t,ll_t,ul_t):
        t = [max(l, min(x, u)) for x, l, u in zip(t, ll_t, ul_t)]
        return t

    def clamp_r(self,r,tcp_center,CONE_TAR,CONE_PHI):
        # caclulate cone_vec: cone_vec is the center vector for the restriction cone regarding CONE_PHI
        cone_vec = CONE_TAR - tcp_center
        cone_vec = cone_vec / np.linalg.norm(cone_vec)
        # calculate alpha
        x_e = np.array([1,0,0])
        q_e = np.quaternion(0,*x_e) # w,x,y,z
        q_t = r*q_e*r.conj()
        x_t = np.array([q_t.x,q_t.y,q_t.z])
        x_t = x_t / np.linalg.norm(x_t)
        alpha = np.arccos(np.dot(cone_vec, x_t))

        if alpha>CONE_PHI:
            # calculate n
            n = np.cross(x_t,cone_vec)
            n = n/np.linalg.norm(n)
            n_x, n_y, n_z = n[0], n[1], n[2]
            alpha_b = alpha - CONE_PHI
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










    def reset(self, obj_pos=None):
        """
        reset to rest poses
        """
        dev = 0.04
        if self.random:
            tcp_tar = obj_pos+np.array([random.uniform(-dev,dev),random.uniform(-dev,dev),0])
            tcp_center = self.TCP_CENTER+np.array([random.uniform(-dev,dev),random.uniform(-dev,dev),random.uniform(-dev,dev)])
        else:
            tcp_tar = self.TCP_TARGET
            tcp_center = self.TCP_CENTER
        tcp_vec = tcp_tar - tcp_center
        tcp_vec = tcp_vec / np.linalg.norm(tcp_vec)
        z_new = self.TCP_UP - np.dot(self.TCP_UP,tcp_vec)*tcp_vec  # z_new = up - proj. of up on tcp_vec
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
        arm_rest_poses = p.calculateInverseKinematics(self.id, self.id_tcp_link, self.TCP_CENTER, orn, jointDamping=self.joints_dampings)
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
        for joint_id in self.joints_controllable_ids:
            pos, vel, _, _ = p.getJointState(self.id, joint_id)
            positions.append(pos)
            velocities.append(vel)
        tcp_pos = p.getLinkState(self.id, self.id_tcp_link)[0]
        return dict(positions=positions, velocities=velocities, tcp_pos=tcp_pos)
    


    def object_is_in_boundaries(self, id_object):
        x, y, _ = p.getBasePositionAndOrientation(id_object)[0]
        return self.LL_T[0] < x < self.UL_T[0] and self.LL_T[1] < y < self.UL_T[1]
