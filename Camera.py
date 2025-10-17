import pybullet as p
import numpy as np


class Camera:
    """
    near = 0.01 # 0.1 means anything closer than 10 cm is invisible
    far = 0.6 # anything further than this is also default fovdefault fov invisible
    """
    def __init__(self, id_robot, id_lens_link, near, far, size, fov):
        self.id_robot = id_robot
        self.id_lens_link = id_lens_link
        self.width, self.height = size
        self.near, self.far = near, far
        self.fov = fov

    def shot(self):
        cam_pos, cam_orn, *_ = p.getLinkState(self.id_robot, self.id_lens_link)
        rot_matrix = np.array(p.getMatrixFromQuaternion(cam_orn)).reshape(3, 3)
        forward = rot_matrix[:, 0]
        up = rot_matrix[:, 2]
        cam_tar = cam_pos+forward

        aspect = self.width / self.height
        self.view_matrix = p.computeViewMatrix(cam_pos, cam_tar, up)
        self.projection_matrix = p.computeProjectionMatrixFOV(self.fov, aspect, self.near, self.far)

        _view_matrix = np.array(self.view_matrix).reshape((4, 4), order='F')
        _projection_matrix = np.array(self.projection_matrix).reshape((4, 4), order='F')
        self.tran_pix_world = np.linalg.inv(_projection_matrix @ _view_matrix)


        # Get depth values using the OpenGL renderer
        _w, _h, rgb, depthImg, seg = p.getCameraImage(self.width, self.height,
                                                   self.view_matrix, self.projection_matrix,
                                                   )
        depth = self.far*self.near/(self.far-(self.far-self.near)*depthImg)
        return rgb, depth, seg

    def approach_stop(self):
        _, depth, _ = self.shot()
        row_9, row_10, row_11 = depth[9], depth[10], depth[11]
        row_9, row_10, row_11 = row_9[5:12], row_10[5:12], row_11[5:12]
        for c in row_9:
            if c<0.0931:
                return True
        for c in row_10:
            if c<0.0945:
                return True
        for c in row_11:
            if c<0.0960:
                return True
        return False