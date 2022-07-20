'''Modules of functions
'''

import numpy as np
import scipy.special as sps
from numpy.typing import ArrayLike


##
def gaussian(
    x: np.ndarray, center: ArrayLike, sigma: float, area: ArrayLike
) -> np.ndarray:
    '''Gaussian function.
    '''
    norm = area / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def reciprocal_exp(r: np.ndarray, norm: float, rnorm: float) -> np.ndarray:
    '''Exponential function
    '''
    return norm * np.exp(-r / rnorm)


def freeman_disk(
    r: np.ndarray, phi: np.ndarray, mass_dyn: float, rnorm: float, incl: float
) -> np.ndarray:
    '''Freeman disk function
    '''
    r2h = 0.5 * r / rnorm
    myu_0 = mass_dyn / (2 * np.pi * rnorm ** 2)

    I0 = sps.i0(r2h)
    K0 = sps.k0(r2h)
    I1 = sps.i1(r2h)
    K1 = sps.k1(r2h)
    if np.any(idx := np.logical_not(np.isfinite(K0))):
        # K0 and K1 become inf at r=0
        K0[idx] = 0.0
        K1[idx] = 0.0
    if np.any(idx := np.logical_not(np.isfinite(I0))):
        # K0 and K1 become inf at r=0
        I0[idx] = 0.0
        I1[idx] = 0.0
    A = I0 * K0 - I1 * K1
    if np.any(idx := (A < 0)):
        A[idx] = 0.0
    f_sightline = np.cos(phi) * np.sin(incl)

    velocity = np.sqrt(4 * np.pi * myu_0 * rnorm * r2h ** 2 * A) * f_sightline
    return velocity
