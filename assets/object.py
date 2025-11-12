import pybullet as p
from pathlib import Path
import pybullet_data
import random

from assets.utilities import random_quaternion


class Object:
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
    