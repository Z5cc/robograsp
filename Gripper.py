import pybullet as p
import math


class Gripper():
    def __init__(self, id, j_names, j_maxForce, j_maxVelocity):
        self.id = id
        self.j_names = j_names
        self.j_maxForce = j_maxForce
        self.j_maxVelocity = j_maxVelocity
        self.gripper_range = [0, 0.085]

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