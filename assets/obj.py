import pybullet as p
from pathlib import Path
import pybullet_data
import random

from assets.util import random_quaternion
from CONSTANTS import LL_T, UL_T


class Obj:
    
    def __init__(self, pos=None, orn=None, index=None):
        self.random = pos is None or orn is None or index is None
        self.index = index
        self.pos = pos
        self.orn = orn

    def reset(self):
        p.removeBody(self.id)
        self.load()
        return self.pos

    def load(self):
        if self.random:
            self.orn = random_quaternion()
            self.pos = [random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),0]
            self.index = random.randint(0,999)
        index_string = str(self.index).zfill(3)
        root = Path(pybullet_data.getDataPath()) / "random_urdfs"
        path = root / index_string / f"{index_string}.urdf"
        path = str(path)
        self.id = p.loadURDF(path,self.pos,self.orn)

    def is_in_boundaries(self):
        x, y, _ = p.getBasePositionAndOrientation(self.id)[0]
        return LL_T[0] < x < UL_T[0] and LL_T[1] < y < UL_T[1]
    