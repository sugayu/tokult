'''Dummy functions
'''

import numpy as np

##

# def no_lensing(coordinate: np.ndarray) -> np.ndarray:
#     '''Dummy function. Return coordinate as itself without lensing.
#     '''
#     return coordinate


def no_lensing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    '''Dummy function. Return coordinate as itself without lensing.'''
    return (x, y)


def no_lensing_interpolation(x: float, y: float) -> np.ndarray:
    '''Dummy function. Return coordinate as itself without lensing.'''
    return np.array([x, y])


def no_convolve(datacube: np.ndarray, index: int = 0) -> np.ndarray:
    '''Dummy function. Return the datacube as itself without convolution.'''
    return datacube
