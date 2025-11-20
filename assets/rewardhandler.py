import pybullet as p
import numpy as np

from CONSTANTS import GAMMA


class RewardHandler:

    def __init__(self, id_robot, id_base_link, id_tcp_link, id_obj, gripper_range):
        self.id_robot = id_robot
        self.id_base_link = id_base_link
        self.id_tcp_link = id_tcp_link
        self.id_obj = id_obj
        self.gripper_range = gripper_range
        self.potential = 0

    def reset(self):
        self.potential = self.get_potential()


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
            # penalty for frequent grasping or penalty for failed grasping
            # r = -1000*self.ray_offset()
            return 0


    # DIFFERENT WAYS TO CALCULATE POTENTIAL
    def successfull_grasp(self):
        lo, hi = p.getAABB(self.id_obj)
        lowest_point_z = lo[2]
        return lowest_point_z>0.005

    def ray_tests(self, ray_start=0.06, ray_length=0.5, graspable_reach=0.07):
        outer_froms, inner_froms = self._get_froms(ray_start)
        outer_tos = self._get_tos(outer_froms, ray_length)
        inner_tos = self._get_tos(inner_froms,ray_length)
        # self._draw_debug_lines(outer_froms,outer_tos)
        # self._draw_debug_lines(inner_froms,inner_tos)

        outer_hits = self._get_hits(outer_froms,outer_tos)
        inner_hits = self._get_hits(inner_froms,inner_tos)
        outer_shortest_hit = self._get_shortest_hit(outer_hits,ray_length)
        inner_shortest_hit = self._get_shortest_hit(inner_hits,ray_length)
        obj_hit = outer_shortest_hit<ray_length or inner_shortest_hit<ray_length
        delta = outer_shortest_hit - inner_shortest_hit
        graspable = inner_shortest_hit < graspable_reach
        p.removeAllUserDebugItems()

        d = max(min(outer_shortest_hit, inner_shortest_hit)-graspable_reach,0)
        return obj_hit, d, delta, graspable
    
    def ray_offset(self):
        # get obj position
        obj_pos, obj_orn = p.getBasePositionAndOrientation(self.id_obj)

        # get straight
        gr_pos, gr_orn, *_ = p.getLinkState(self.id_robot,self.id_tcp_link)
        rot_matrix = np.array(p.getMatrixFromQuaternion(gr_orn)).reshape(3, 3)
        gr_forw = rot_matrix[:,0]

        obj_pos, gr_pos, gr_forw = map(np.array,(obj_pos, gr_pos, gr_forw))
        cross = np.cross(gr_forw, obj_pos-gr_pos)
        offset = float(np.linalg.norm(cross)/np.linalg.norm(gr_forw))
        # print(f'offset:{offset}')
        return offset


    # HELPER FOR POTENTIAL CALCULATION
    def _draw_debug_lines(self, froms, tos, visible=True):
        if visible:
            for f, to in zip(froms, tos):
                p.addUserDebugLine(f, to, [0,1,0], lineWidth=1.5)
                
    def _get_froms(self, ray_start):
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
        inner_frame = list(map(self._local_to_global, inner_frame))
        right_outer_frame = list(map(self._local_to_global, right_outer_frame))
        left_outer_frame = list(map(self._local_to_global, left_outer_frame))
        inner_grid = self._interpolate_grid(*inner_frame,3,5)
        right_outer_grid = self._interpolate_grid(*right_outer_frame,2,3)
        left_outer_grid = self._interpolate_grid(*left_outer_frame,2,3)
        outer_grid = np.concatenate([left_outer_grid, right_outer_grid], axis=0)

        inner_froms = inner_grid.reshape(-1,3).tolist()
        outer_froms = outer_grid.reshape(-1,3).tolist()
        return outer_froms, inner_froms
    
    def _get_tos(self,froms, ray_length):
        pos, orn, *_ = p.getLinkState(self.id_robot,self.id_base_link)
        rot_matrix = np.array(p.getMatrixFromQuaternion(orn)).reshape(3, 3)
        forward = rot_matrix[:, 2]
        tos = [f + forward*ray_length for f in froms]
        return tos
    
    def _get_hits(self, froms, tos):
        hits = p.rayTestBatch(froms, tos)
        return hits

    def _get_shortest_hit(self, hits, ray_length):
        absolute_fractions = (
            hit[2] * ray_length if hit[0] == self.id_obj else ray_length
            for hit in hits
        )
        return min(absolute_fractions)


    # HELPER FOR HELPER
    def _local_to_global(self, pt_local):
        pos, orn, *_ = p.getLinkState(self.id_robot,self.id_base_link)
        pt_world, _ = p.multiplyTransforms(pos, orn, pt_local, (0,0,0,1))
        return pt_world

    def _interpolate_grid(self, v00, v10, v01, v11, ni, nj):
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
    