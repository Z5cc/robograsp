import pybullet as p
import numpy as np
import random
from scipy.spatial.transform import Rotation as R

from CONSTANTS import CONE_CENTER, NUMERIC_DAMPING, BASE_POS, BASE_ORN, TCP_UP
from assets.util import target_from_delta_to_world, target_from_world_to_tcp
from assets.gripper import Gripper


class Joint():
    
    def __init__(self, index, joint_type, max_force, max_vel, controllable):
        self.index = index
        self.joint_type = joint_type
        self.max_force = max_force
        self.max_vel = max_vel
        self.controllable = controllable


class Robot:

    def __init__(self):
        self.id = p.loadURDF('./ur5_robotiq_85/urdf/ur5_robotiq_85.urdf', BASE_POS, BASE_ORN,
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
            self.joints_dampings.append(NUMERIC_DAMPING)

            self.link_map[link_name] = index
            self.joint_map[joint_name] = index
            self.joints[index] = joint

            if controllable:
                self.joints_controllable_ids.append(index)
                if len(self.joints_controllable_ids)<=self.arm_num_dofs:
                    self.joints_controllable_arm_ids.append(index)

        for joint_id in self.joints_controllable_ids:
            p.setJointMotorControl2(self.id, joint_id, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        
        self.gripper = Gripper(self.id, self.link_map ,self.joint_map, self.joints)

    def reset(self, tcp_center, tcp_target):
        tcp_vec = tcp_target - tcp_center
        tcp_vec = tcp_vec / np.linalg.norm(tcp_vec)
        z_new = TCP_UP - np.dot(TCP_UP,tcp_vec)*tcp_vec  # z_new = up - proj. of up on tcp_vec
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
        arm_rest_poses = p.calculateInverseKinematics(self.id, self.link_map['tcp_link'], tcp_center, orn, jointDamping=self.joints_dampings)
        for rest_pose, joint_id in zip(arm_rest_poses, self.joints_controllable_arm_ids):
            p.resetJointState(self.id, joint_id, rest_pose)
        # 4. drive motors to reseted joint states to hold new position
        for rest_pose, joint_id in zip(arm_rest_poses, self.joints_controllable_arm_ids):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, rest_pose,
                            force=self.joints[joint_id].max_force, maxVelocity=self.joints[joint_id].max_vel)        
        self.gripper.reset() # drive gripper to default open position
        # 5. let world settle -> done in env, because besides robot obj also needs to settle

    def get_gripper_range(self):
        return self.gripper.gripper_range
    
    def get_link_pos(self, link):
        link_id = self.link_map[link]
        pos, orn, *_ = p.getLinkState(self.id, link_id)
        return pos, orn # translation t (x,y,z) and quaternion r (x,y,z,w)


    # FUNCTIONS FOR MOVING TCP AND GRIPPER
    def move_tcp_delta(self, delta):
        t, r = self.get_link_pos('tcp_link')
        t, r = target_from_delta_to_world(t, r, delta)
        self.move_tcp_abs(t, r)

    def move_tcp_abs(self, t, r):            
        joint_poses = p.calculateInverseKinematics(self.id, self.link_map['tcp_link'], t, r, jointDamping=self.joints_dampings)
        # arm
        for joint_pose, joint_id in zip(joint_poses, self.joints_controllable_arm_ids):
            p.setJointMotorControl2(self.id, joint_id, p.POSITION_CONTROL, joint_pose,
                                    force=self.joints[joint_id].max_force, maxVelocity=self.joints[joint_id].max_vel)
            
    def open_gripper(self):
        self.gripper.open()

    def close_gripper(self):
        self.gripper.close()
            
    def move_gripper(self, open_length):
        self.gripper.move(open_length)
    

    # FUNCTIONS FOR ACTIONS
    # ACTION 0
    def grasp(self):
        yield from self.approach()
        liftable = yield from self.close()
        if liftable:
            yield from self.lift()
        else:
            yield from self.retreat()

    # ACTION 1 - ...
    def approach(self):
        dx = 0.005
        delta = [dx,0,0,0,0,0]
        x_approach_stop = False
        while not (x_approach_stop):
            t, r = self.get_link_pos('tcp_link')
            x_old = target_from_world_to_tcp(t,r)[0]
            self.move_tcp_delta(delta)
            yield 30
            t, r = self.get_link_pos('tcp_link')
            x = target_from_world_to_tcp(t,r)[0]
            x_approach_stop = x_old+0.9*dx > x # if x does not reach the goal of x_old+0.9*dx

    def close(self):
        liftable=False
        self.close_gripper()
        c,i=0,0
        while (not self.gripper.gr_closed()) and (i<100):
            self.gripper.save_angle()
            yield 60
            c=c+1 if self.gripper.has_obj(include_delta=True) else 0
            i=i+1
            # print(i)
            if c==4:
                liftable=True
                break
        return liftable

    def lift(self):
        t,r = self.get_link_pos('tcp_link')
        t = list(t)
        while t[2]<0.2: # lift in z direction
            t[2]+=0.01
            self.move_tcp_abs(t, r)
            yield 30
            if not self.gripper.has_obj(include_delta=False):
                yield from self.retreat()
                break

    def retreat(self):
        self.open_gripper()
        delta = [-0.01,0,0,0,0,0]
        for _ in range(5):
            self.move_tcp_delta(delta)
            yield 30

    def seek(self,action):
        # default deltas
        dt = 0.015
        dr = 0.05
        # [dx,dy,dz,droll,dpitch,dyaw]
        delta_lookup = {
            1:  [ dt, 0 , 0 , 0 , 0 , 0 ],
            2:  [-dt, 0 , 0 , 0 , 0 , 0 ],
            3:  [ 0 , dt, 0 , 0 , 0 , 0 ],
            4:  [ 0 ,-dt, 0 , 0 , 0 , 0 ],
            5:  [ 0 , 0 , dt, 0 , 0 , 0 ],
            6:  [ 0 , 0 ,-dt, 0 , 0 , 0 ],
            7:  [ 0 , 0 ,  0, dr, 0 , 0 ],
            8:  [ 0 , 0 ,  0,-dr, 0 , 0 ],
            9:  [ 0 , 0 ,  0, 0 , dr, 0 ],
            10: [ 0 , 0 ,  0, 0 ,-dr, 0 ],
            11: [ 0 , 0 ,  0, 0 , 0 , dr],
            12: [ 0 , 0 ,  0, 0 , 0 ,-dr],
        }
        delta = delta_lookup[action]
        # move arm and gripper
        self.move_tcp_delta(delta)
        self.open_gripper()
        yield 30
