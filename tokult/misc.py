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
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos)


def polar_coord(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Convert (x, y) to polar coordinates (r, phi)
    '''
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi


def fft2(cube: np.ndarray, zero_padding: bool = False) -> np.ndarray:
    '''2 dimensional Fourier transform.
    '''
    shift = (np.array(cube.shape[1:]) / 2.0).astype(int)
    cube_shift = np.roll(cube, shift, axis=(1, 2))
    uvcube = np.fft.fft2(cube_shift)
    uvcube = np.fft.fftshift(uvcube, axes=(1, 2))
    return uvcube


def ifft2(uvcube: np.ndarray) -> np.ndarray:
    '''Inverse 2 dimensional Fourier transform.
    '''
    cube_shift = np.fft.ifftshift(uvcube, axes=(1, 2))
    cube_shift = np.fft.ifft2(cube_shift)
    shift = -(np.array(cube_shift.shape[1:]) / 2.0).astype(int)
    cube = np.roll(cube_shift, shift, axis=(1, 2))
    return cube


def no_lensing(coordinate: np.ndarray) -> np.ndarray:
    '''Dummy function. Return coordinate as itself without lensing.
    '''
    return coordinate


def no_convolve(datacube: np.ndarray, index: int = 0) -> np.ndarray:
    '''Dummy function. Return the datacube as itself without convolution.
    '''
    return datacube
