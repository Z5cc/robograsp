import pybullet as p
from pathlib import Path
import pybullet_data
import random

from constants import LL_T, UL_T


class Obj:
    
    def __init__(self):
        self.id = None

    def reset(self, pos, orn, index):
        index_string = str(index).zfill(3)
        root = Path(pybullet_data.getDataPath()) / "random_urdfs"
        path = root / index_string / f"{index_string}.urdf"
        path = str(path)
        if self.id:
            p.removeBody(self.id)
        self.id = p.loadURDF(path,pos,orn)
        return self.id

    def is_in_boundaries(self):
        x, y, _ = p.getBasePositionAndOrientation(self.id)[0]
        return LL_T[0] < x < UL_T[0] and LL_T[1] < y < UL_T[1]

    def get_pos(self):
        pos, orn = p.getBasePositionAndOrientation(self.id)
        return pos, orn

    def get_AABB(self):
        lo, hi = p.getAABB(self.id)
        return lo, hi
