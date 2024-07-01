'''Utilities to convert coordinates.
'''

import numpy as np


##
def to_object_from(
    coord_celestial: np.ndarray, PA: float, incl: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Convert coordinates from celestial coordinates to object polar coordinates.'''
    pa = PA
    inclination = incl

    # coord_source = lensing(coord_image)
    coord_object = rotate(coord_celestial, pa)
    xx, yy = np.moveaxis(coord_object, -1, 0)
    yy = yy / np.cos(inclination)
    r, phi = polar(xx, yy)
    return r, phi


def to_relative_from(
    coord_source: np.ndarray, at_x0: float, at_y0: float
) -> np.ndarray:
    '''Convert coordinates from absolute positions to relative positions.'''
    global lensing_interpolation
    central_position = lensing_interpolation(at_x0, at_y0)
    return coord_source - central_position[np.newaxis, np.newaxis, :]


def rotate(pos: np.ndarray, angle: float) -> np.ndarray:
    '''Rotate (x,y) coordinates
    Keyword Arguments:
    pos -- position array. shape: (n, m, 2)
    angle -- scalar; angle to rotate. radian
    '''
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos, -1)


def polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Convert (x, y) to polar coordinates (r, phi)'''
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi
