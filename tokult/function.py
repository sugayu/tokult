'''Modules of functions
'''
from __future__ import annotations
import numpy as np
import scipy.special as sps
from typing import Union
from numpy.typing import ArrayLike
from astropy import units as u
import astropy.constants as const


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
    myu_0_norm = mass_dyn / rnorm

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

    velocity = r2h * np.sqrt(2 * myu_0_norm * A) * f_sightline
    return velocity


def maximum_rotation_velocity(
    mass_dyn: Union[u.Quantity, np.ndarray, float],
    rnorm: Union[u.Quantity, np.ndarray, float],
) -> Union[u.Quantity, np.ndarray, float]:
    '''Maximum rotation velocity of Freeman disk function
    '''
    if isinstance(mass_dyn, u.Quantity):
        _mass_dyn = const.G * mass_dyn.to(u.kg)
    elif isinstance(mass_dyn, np.ndarray) or isinstance(mass_dyn, float):
        _mass_dyn = 10.0 ** mass_dyn

    if isinstance(rnorm, u.Quantity):
        _rnorm = rnorm.to(u.km)
    elif isinstance(rnorm, np.ndarray) or isinstance(rnorm, float):
        _rnorm = rnorm

    vmax = 0.88 * np.sqrt(_mass_dyn / (2 * _rnorm))
    if isinstance(vmax, u.Quantity):
        vmax = vmax.decompose().to(u.km / u.s)
    return vmax
