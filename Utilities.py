import pybullet as p
import glob
from collections import namedtuple
from attrdict import AttrDict
import functools
import torch
import cv2
from scipy import ndimage
import numpy as np
from pathlib import Path
import random





class Object:
    def __init__(self, pos):
        self.files = ['clear_box','green_cup','green_bowl']
        self.pos = pos

    def load(self):
        object = random.choice(self.files)
        path = Path("urdf_objects") / object / "model.urdf"
        path = str(path)
        self.id = p.loadURDF(path,self.pos)
    
    def reset(self):
        p.removeBody(self.id)
        self.load()




class Camera:
    def __init__(self, cam_pos, cam_tar, cam_up, near, far, size, fov):
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, cam_up)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

    def shot(self):
        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depth, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        return rgb, depth, seg
