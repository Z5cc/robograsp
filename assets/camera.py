import pybullet as p
import numpy as np

from CONSTANTS import H,W


class Camera:
    """
    near = 0.01 # 0.1 means anything closer than 10 cm is invisible
    far = 0.6 # anything further than this is also default fovdefault fov invisible
    """
    def __init__(self, id_robot, id_lens_link):
        self.id_robot = id_robot
        self.id_lens_link = id_lens_link
        self.NEAR, self.FAR = 0.01, 5
        self.FOV = 50

    def shot(self):
        cam_pos, cam_orn, *_ = p.getLinkState(self.id_robot, self.id_lens_link)
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_matrix[:, 0]
        up = rot_matrix[:, 2]
        cam_tar = cam_pos+forward

        aspect = W / H
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, up)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.FOV, aspect, self.NEAR, self.FAR)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)

        # Get depth values using the OpenGL renderer
        _w, _h, _, depthImg, _ = p.getCameraImage(W, H,
                                                   self.view_matrix, self.projection_matrix,flags=p.ER_NO_SEGMENTATION_MASK
                                                   )
        depth = self.FAR*self.NEAR/(self.FAR-(self.FAR-self.NEAR)*depthImg)
        return depth
