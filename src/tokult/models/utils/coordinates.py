'''Utilities to convert coordinates.
'''

from typing import Callable
import numpy as np


##
lensing_interpolation: Callable


def to_object_from(
    coord_celestial: np.ndarray, PA: float, incl: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Convert coordinates from celestial coordinates to object polar coordinates.

    Args:
        coord_celestial (np.ndarray): Celestial coordinates with shape of (ny, nx, 2).
        PA (float): Position angle in a unit of radian.
        incl (float): Inclination in a unit of radian.

    Returns:
        tuple[np.ndarray, np.ndarray]: Polar coordinates on the object.
    '''

    pa = PA
    inclination = incl

    # coord_source = lensing(coord_image)

    # The negative sign of -pa is to make the major axis to the x axis.
    coord_object = rotate(coord_celestial, -pa)
    yy, xx = np.moveaxis(coord_object, -1, 0)
    yy = yy / np.cos(inclination)
    r, phi = polar(xx, yy)
    return r, phi


def to_relative_from(
    coord_source: np.ndarray, at_x0: float, at_y0: float
) -> np.ndarray:
    '''Convert coordinates from absolute positions to relative positions.

    Args:
        coord_source (np.ndarray): Absolute coordinates. The expected shape is
            (ny, nx, 2).
        at_x0 (float): Central x position.
        at_y0 (float): Central y position.

    Returns:
        np.ndarray: Coordinates relative to the galaxy center The expected
            shape is (ny, nx, 2).
    '''
    global lensing_interpolation
    try:
        central_position = lensing_interpolation(at_x0, at_y0)
    except NameError:
        central_position = np.array((at_y0, at_x0))
    return coord_source - central_position[np.newaxis, np.newaxis, :]


def rotate(pos: np.ndarray, angle: float) -> np.ndarray:
    '''Anticlockwise rotatation of (y, x) coordinates.

    It changes a coordinate of (y,x)=(0,1) into (1,0).

    Args:
        pos (np.ndarray): Positional array. Shape: (ny, nx, 2)
        angle (float): Scalar; Angle [radian] to rotate.

    Returns:
        np.ndarray: Rotated coordinates. Shape: (ny, nx, 2)
    '''
    assert len(pos.shape) == 3
    assert pos.shape[2] == 2
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos, -1)


def polar(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Convert (x, y) to polar coordinates (r, phi)'''
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi
