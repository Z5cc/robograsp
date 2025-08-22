import pybullet as p
import math
import numpy as np
from collections import namedtuple




class Gripper():
    def __init__(self, id, base_link_id, j_names, j_maxForce, j_maxVelocity, object, max_open=0.05):
        self.id = id
        self.base_link_id = base_link_id
        self.j_names = j_names
        self.j_maxForce = j_maxForce
        self.j_maxVelocity = j_maxVelocity
        self.object = object
        self.gripper_range = [0, 0.085 if max_open>0.085 else max_open]

        # To control the gripper
        mimic_parent_name = 'finger_joint'
        mimic_children_names = {'right_outer_knuckle_joint': 1,
                                'left_inner_knuckle_joint': 1,
                                'right_inner_knuckle_joint': 1,
                                'left_inner_finger_joint': -1,
                                'right_inner_finger_joint': -1}
        self.__setup_mimic_joints__(mimic_parent_name, mimic_children_names)
        
        
    def __setup_mimic_joints__(self, mimic_parent_name, mimic_children_names):
        self.mimic_parent_id = [joint_id for joint_id, name in enumerate(self.j_names) if name == mimic_parent_name][0]
        self.mimic_child_multiplier = {joint_id: mimic_children_names[name] for joint_id, name in enumerate(self.j_names) if name in mimic_children_names}

        for joint_id, multiplier in self.mimic_child_multiplier.items():
            c = p.createConstraint(self.id, self.mimic_parent_id,
                                   self.id, joint_id,
                                   jointType=p.JOINT_GEAR,
                                   jointAxis=[0, 1, 0],
                                   parentFramePosition=[0, 0, 0],
                                   childFramePosition=[0, 0, 0])
            p.changeConstraint(c, gearRatio=-multiplier, maxForce=100, erp=1)  # Note: the mysterious `erp` is of EXTREME importance


    def move(self, open_length):
        # open_length = np.clip(open_length, *self.gripper_range)
        open_angle = 0.715 - math.asin((open_length - 0.010) / 0.1143)  # angle calculation
        # Control the mimic gripper joint(s)
        p.setJointMotorControl2(self.id, self.mimic_parent_id, p.POSITION_CONTROL, targetPosition=open_angle,
                                force=self.j_maxForce[self.mimic_parent_id], maxVelocity=self.j_maxVelocity[self.mimic_parent_id])
        

    def reset(self):
        self.open()

    def open(self):
        self.move(self.gripper_range[1])

    def close(self):
        self.move(self.gripper_range[0])


    def get_opening_length(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        open_angle = joint_state[0]
        open_length = 0.010 + 0.1143*math.sin(0.715-open_angle)
        return open_length
    
    def get_velocity(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        return joint_state[1]
    
    def get_torque(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        return joint_state[3]

    def has_object(self,threshold=1):
        gripper_torque = self.get_torque()
        if gripper_torque>threshold:
            return True
        else:
            return False
        
    





    def center_test(self):
        test_cube = 
        return p.intersection(test_cube, self.object.id)

    def interpolate_grid(v00, v10, v01, v11, ni, nj):
        v00, v10, v01, v11 = map(np.array, [v00, v10, v01, v11])
        u = np.linspace(0, 1, ni)
        v = np.linspace(0, 1, nj)
        uu, vv = np.meshgrid(u, v, indexing='ij') # i rows, j columns
        # bilinear interpolation formula
        points = ((1-uu)*(1-vv))[:, :, None, None]*v00 \
               + (uu*(1-vv))[:, :, None, None]*v10 \
               + ((1-uu)*vv)[:, :, None, None]*v01 \
               + (uu*vv)[:, :, None, None]*v11
        return points


    def get_froms(self):
        pos, orn, *_ = p.getLinkState(self.base_link_id)

        return outer_froms, inner_froms    

    
    def get_tos(self,froms, ray_length):
        pos, orn, *_ = p.getLinkState(self.base_link_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot_matrix[:, 0]
        tos = [f + forward*ray_length for f in froms]
        return tos
    
    def get_hits(self, froms, tos):
        hits = p.rayTestBatch(self,froms, tos)
        return hits

    
    def object_is_hit(self, hits):
        for hit in hits:
            if hit[0]==self.object.id:
                return True
            
    def get_shortest_hit(self, hits, ray_length):
        x = ray_length
        for hit in hits:
            absolute_fraction=ray_length
            if hit[0]==self.object.id:
                absolute_fraction = hit[2]*ray_length
            if absolute_fraction<ray_length:
                x = absolute_fraction
        return x






    def get_delta_outer_inner_rays(self,ray_length):
        outer_froms = 
        outer_tos = self.get_tos(outer_froms, ray_length)
        inner_froms = 
        inner_tos = self.get_tos(inner_froms,ray_length)

        outer_hits = self.get_hits(outer_froms,outer_tos)
        inner_hits = self.get_hits(inner_froms,inner_tos)
        return self.get_shortest_hit(self,outer_hits,ray_length) - self.get_shortest_hit(self,inner_hits,ray_length)
