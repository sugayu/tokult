'''miscellaneous functions
'''
import numpy as np


##
def rotate_coord(pos: np.ndarray, angle: float) -> np.ndarray:
    '''Rotate (x,y) coordinates
    Keyword Arguments:
    pos -- position array. shape: (n, m, 2)
    angle -- scalar; angle to rotate. radian
    '''
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos)


def polar_coord(x, y):
    '''Convert (x, y) to polar coordinates (r, phi)
    '''
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi
