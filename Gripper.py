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
    
    def get_angle(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        return joint_state[0]
    
    def get_velocity(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        return joint_state[1]
    
    def get_torque(self):
        joint_state = p.getJointState(self.id, self.mimic_parent_id)
        return joint_state[3]

    # regarding angle_THOLD: 0.776rad are 0.003m in absolute, smaller angle -> bigger open_width
    # regarding delta_THOLD: 0.01rad are equivalent to 0.001m in relative
    def has_object(self, torque_THOLD=2, angle_THOLD=0.776):
        torque = self.get_torque()
        angle = self.get_angle()
        return torque>torque_THOLD and angle<angle_THOLD # and delta<delta_THOLD
    
    def gr_closed(self, angle_THOLD=0.776):
        return self.get_angle()>angle_THOLD # 0.8rad is about 0.001m opening length
        
    






    def local_to_global(self, pt_local):
        pos, orn, *_ = p.getLinkState(self.id,self.base_link_id)
        pt_world, _ = p.multiplyTransforms(pos, orn, pt_local, (0,0,0,1))
        return pt_world

    def interpolate_grid(self, v00, v10, v01, v11, ni, nj):
        v00, v10, v01, v11 = map(np.array, [v00, v10, v01, v11])
        u = np.linspace(0, 1, ni)
        v = np.linspace(0, 1, nj)
        uu, vv = np.meshgrid(u, v, indexing='ij') # i rows, j columns
        # bilinear interpolation formula
        grid = ((1-uu)*(1-vv))[:, :, None, None]*v00 \
               + (uu*(1-vv))[:, :, None, None]*v10 \
               + ((1-uu)*vv)[:, :, None, None]*v01 \
               + (uu*vv)[:, :, None, None]*v11
        return grid

    def get_froms(self, ray_start):
        a = ray_start
        h = 0.011
        w = self.gripper_range[1]/2
        iw = w-0.002
        ih = h-0.001
        ow = w-0.001
        oow = w+0.0065+0.009+0.001
        oh = h+0.001

        inner_frame = [(ih,-iw,a),(-ih,-iw,a),( ih, iw,a),(-ih, iw,a)] # top_left, bottom_left, top_right, bottom_right like v00, v10, v01, v11
        right_outer_frame = [( oh, ow,a),(-oh, ow,a),( oh,oow,a),(-oh,oow,a)] # top_left, bottom_left, top_right, bottom_right like v00, v10, v01, v11
        left_outer_frame = [(x,-y,z) for (x,y,z) in right_outer_frame] # mirror right_outer_frame
        inner_frame = list(map(self.local_to_global, inner_frame))
        right_outer_frame = list(map(self.local_to_global, right_outer_frame))
        left_outer_frame = list(map(self.local_to_global, left_outer_frame))
        inner_grid = self.interpolate_grid(*inner_frame,3,5)
        right_outer_grid = self.interpolate_grid(*right_outer_frame,2,3)
        left_outer_grid = self.interpolate_grid(*left_outer_frame,2,3)
        outer_grid = np.concatenate([left_outer_grid, right_outer_grid], axis=0)

        inner_froms = inner_grid.reshape(-1,3).tolist()
        outer_froms = outer_grid.reshape(-1,3).tolist()
        return outer_froms, inner_froms
    
    def get_tos(self,froms, ray_length):
        pos, orn, *_ = p.getLinkState(self.id,self.base_link_id)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot_matrix[:, 2]
        tos = [f + forward*ray_length for f in froms]
        return tos
    







    def get_hits(self, froms, tos):
        hits = p.rayTestBatch(froms, tos)
        return hits

    def get_shortest_hit(self, hits, ray_length):
        absolute_fractions = (
            hit[2] * ray_length if hit[0] == self.object.id else ray_length
            for hit in hits
        )
        return min(absolute_fractions)




    def draw_debug_lines(self, froms, tos, visible=True):
        if visible:
            for f, to in zip(froms, tos):
                p.addUserDebugLine(f, to, [0,1,0], lineWidth=1.5)


    def ray_tests(self, ray_start=0.06, ray_length=0.5, graspable_reach=0.07):
        outer_froms, inner_froms = self.get_froms(ray_start)
        outer_tos = self.get_tos(outer_froms, ray_length)
        inner_tos = self.get_tos(inner_froms,ray_length)
        # self.draw_debug_lines(outer_froms,outer_tos)
        # self.draw_debug_lines(inner_froms,inner_tos)

        outer_hits = self.get_hits(outer_froms,outer_tos)
        inner_hits = self.get_hits(inner_froms,inner_tos)
        outer_shortest_hit = self.get_shortest_hit(outer_hits,ray_length)
        inner_shortest_hit = self.get_shortest_hit(inner_hits,ray_length)
        object_hit = outer_shortest_hit<ray_length or inner_shortest_hit<ray_length
        delta = outer_shortest_hit - inner_shortest_hit
        graspable = inner_shortest_hit < graspable_reach
        p.removeAllUserDebugItems()
        return object_hit, delta, graspable
    