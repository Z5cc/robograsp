import random
import numpy as np


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
