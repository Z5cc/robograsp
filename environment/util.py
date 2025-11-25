import random
import numpy as np
import quaternion
import pybullet as p

from CONSTANTS import CONE_CENTER, CONE_TAR, CONE_PHI, LL_T, UL_T


def target_from_delta_to_world(t, r, delta):
    """
    delta is first converted from tcp to world orientation.
    then this new oriented delta is added to t and r,
    which results in new absolute targets t and r.
    t: (x,y,z)
    r: (x,y,z,w)
    delta: [dx,dy,dz,droll,dpitch,dyaw]
    """
    # 1. converting to numpy
    t, r = np.array(t), np.array(r)
    dt_TCP = np.array(delta[0:3])
    dr_TCP = np.array(p.getQuaternionFromEuler(delta[3:6])) # one rotation dr_TCP derived from intrinsic euler angles
    # 2. translation
    R_TCP_to_W = _get_R_TCP_to_W(r)
    dt = R_TCP_to_W @ dt_TCP # from local to global world frame
    t += dt
    # 3. rotation
    r = np.quaternion(r[3],r[0],r[1],r[2]) # pybullet quaternion: xyzw  numpy quaternion: wxyz
    dr_TCP = np.quaternion(dr_TCP[3],dr_TCP[0],dr_TCP[1],dr_TCP[2])
    r = r * dr_TCP # do not need to multiplicate with individual like = r*dr_yaw*dr_pitch*dr_roll, because getQuaternionFromEuler is from intrinsic angles

    t = t.tolist()
    t, r = _clamp_t(t), _clamp_r(r)
    r = [r.x,r.y,r.z,r.w] # convert back to pybullet convention
    return t, r


def target_from_world_to_tcp(t, r):
    """
    convert t from world to tcp orientation
    """
    t, r = np.array(t), np.array(r)
    R_W_to_TCP = _get_R_W_to_TCP(r)
    t = R_W_to_TCP @ t
    return t


def random_quaternion():
    """
    return a uniform random quaternion in format [x,y,z,w]
    according to the 'subgroup algorithm' in
    graphics gems III: https://theswissbay.ch/pdf/GentoomenRandom%20Library/Game%20Development/Programming/Graphics%20Gems%203.pdf
    original C implementation: https://github.com/erich666/GraphicsGems/blob/master/gemsiii/urot.c
    """
    x0,x1,x2 = random.random(), random.random(), random.random()
    s1 = np.sin(2*np.pi*x1)
    c1 = np.cos(2*np.pi*x1)
    s2 = np.sin(2*np.pi*x2)
    c2 = np.cos(2*np.pi*x2)
    r1 = np.sqrt(1-x0)
    r2 = np.sqrt(x0)
    return [s1*r1, c1*r1, s2*r2, c2*r2]


def _get_R_TCP_to_W(r):
    R_TCP_to_W = np.array(p.getMatrixFromQuaternion(r)).reshape(3, 3)
    return R_TCP_to_W

def _get_R_W_to_TCP(r):
    R_TCP_to_W = _get_R_TCP_to_W(r)
    R_W_to_TCP = np.transpose(R_TCP_to_W)
    return R_W_to_TCP

def _clamp_t(t):
    t = [max(l, min(x, u)) for x, l, u in zip(t, LL_T, UL_T)]
    return t

def _clamp_r(r):
    # caclulate cone_vec: cone_vec is the center vector for the restriction cone regarding CONE_PHI
    cone_vec = CONE_TAR - CONE_CENTER
    cone_vec = cone_vec / np.linalg.norm(cone_vec)
    # calculate alpha
    x_e = np.array([1,0,0])
    q_e = np.quaternion(0,*x_e) # w,x,y,z
    q_t = r*q_e*r.conj()
    x_t = np.array([q_t.x,q_t.y,q_t.z])
    x_t = x_t / np.linalg.norm(x_t)
    alpha = np.arccos(np.dot(cone_vec, x_t))

    if alpha>CONE_PHI:
        # calculate n
        n = np.cross(x_t,cone_vec)
        n = n/np.linalg.norm(n)
        n_x, n_y, n_z = n[0], n[1], n[2]
        alpha_b = alpha - CONE_PHI
        sin_half = np.sin(alpha_b/2)
        cos_half = np.cos(alpha_b/2)
        r_back = np.quaternion(cos_half, sin_half*n_x, sin_half*n_y, sin_half*n_z)
        r = r_back*r
        return r
    return r
