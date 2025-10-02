import pybullet as p
from pathlib import Path
import pybullet_data
import random
from Utilities import random_quaternion


class Object:
    def __init__(self, ll, ul):
        # self.files = ['clear_box','green_cup','green_bowl']
        self.ll = ll
        self.ul = ul
        self.load()

    def load(self):
        self.orn = random_quaternion()
        self.pos = [random.uniform(-0.1,0.1),random.uniform(-0.1,0.1),0]
        object = str(random.randint(0,999)).zfill(3)
        root = Path(pybullet_data.getDataPath()) / "random_urdfs"
        path = root / object / f"{object}.urdf"
        path = str(path)
        self.id = p.loadURDF(path,self.pos,self.orn)
    
    def reset(self):
        p.removeBody(self.id)
        self.load()
        return self.pos


    def is_in_boundaries(self):
        x, y, _ = p.getBasePositionAndOrientation(self.id)[0]
        return self.ll[0] < x < self.ul[0] and self.ll[1] < y < self.ul[1]
    