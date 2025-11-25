import pybullet as p
import numpy as np

from CONSTANTS import GAMMA


class RewardHandler:

    def __init__(self, robot, obj):
        self.robot = robot
        self.obj = obj
        self.potential = 0
        self.offset_line_id = None
        self.point_id = None
        self.vis = False

    def reset(self):
        self.potential = self.get_potential()
        if self.vis is True:
            p.removeUserDebugItem(self.offset_line_id)
            p.removeUserDebugItem(self.point_id)
            self.offset_line_id = None
            self.point_id = None


    # RETURN REWARD
    def get_reward(self):
        next_potential = self.get_potential()
        r = GAMMA*next_potential-self.potential
        self.potential = next_potential
        return r

    def get_potential(self):
        if self.successfull_grasp():
            return 100
        else:
            # r = -1000*self.ray_offset()
            return 0


    # DIFFERENT WAYS TO CALCULATE POTENTIAL
    def successfull_grasp(self):
        lo, hi = self.obj.get_AABB()
        lowest_point_z = lo[2]
        return lowest_point_z>0.005
    
    def ray_offset(self):
        # get obj position
        obj_pos, obj_orn = self.obj.get_pos()

        # get straight
        gr_pos, gr_orn, *_ = self.robot.get_link_pos('robotiq_arg2f_base_link')
        rot_matrix = np.array(p.getMatrixFromQuaternion(gr_orn)).reshape(3, 3)
        gr_forw = rot_matrix[:,2]
        if self.vis is True:
            self._draw_debug_line(gr_pos, gr_pos+gr_forw)
            self._draw_point(obj_pos)

        obj_pos, gr_pos, gr_forw = map(np.array,(obj_pos, gr_pos, gr_forw))
        cross = np.cross(gr_forw, obj_pos-gr_pos)
        offset = float(np.linalg.norm(cross)/np.linalg.norm(gr_forw))
        return offset

    def _draw_debug_line(self, f, to):
        if self.offset_line_id is None:
            self.offset_line_id = p.addUserDebugLine(f, to, [0,1,0], lineWidth=1.5)
        else:
            p.addUserDebugLine(f, to, [0,1,0], lineWidth=1.5, replaceItemUniqueId=self.offset_line_id)

    def _draw_point(self, pos):
        if self.point_id is None:
            self.point_id = p.addUserDebugPoints([pos], [[1,0,0]], pointSize=6)
        else:
            p.addUserDebugPoints([pos], [[1,0,0]], pointSize=8, replaceItemUniqueId=self.point_id)
