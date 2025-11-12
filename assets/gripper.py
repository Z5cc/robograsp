import pybullet as p
import math



class Gripper():
    def __init__(self, id, link_map, joint_map, joints, max_open=0.05):
        self.id = id
        self.link_map = link_map
        self.joint_map = joint_map
        self.joints = joints
        self.id_base_link = link_map['robotiq_arg2f_base_link']
        self.joints = joints
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
        self.mimic_parent_id = self.joint_map[mimic_parent_name]
        self.mimic_child_multiplier = {self.joint_map[name]: mimic for name, mimic in mimic_children_names.items()}

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
                                force=self.joints[self.mimic_parent_id].max_force, maxVelocity=self.joints[self.mimic_parent_id].max_vel)
        

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
    
    def save_angle(self):
        self.old_angle = self.get_angle()

    # regarding angle_THOLD: 0.776rad are 0.003m in absolute, smaller angle -> bigger open_width
    # regarding delta_THOLD: 0.01rad are equivalent to 0.001m in relative
    def has_obj(self, include_delta, torque_THOLD=2, angle_THOLD=0.776, delta_THOLD=0.01):
        torque = self.get_torque()
        angle = self.get_angle()
        delta = self.old_angle-angle
        if include_delta:
            return torque>torque_THOLD and angle<angle_THOLD and abs(delta)<delta_THOLD
        else:
            return torque>torque_THOLD and angle<angle_THOLD
    
    def gr_closed(self, angle_THOLD=0.776):
        return self.get_angle()>angle_THOLD # 0.8rad is about 0.001m opening length
