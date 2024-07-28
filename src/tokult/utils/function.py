'''Modules of functions

Should be deleted.
'''

from __future__ import annotations
import numpy as np
from typing import Union
from numpy.typing import ArrayLike
from astropy import units as u
import astropy.constants as const


##
def gaussian(
    x: np.ndarray, center: ArrayLike, sigma: float, area: ArrayLike
) -> np.ndarray:
    '''Gaussian function.'''
    norm = area / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - center) ** 2) / (2.0 * sigma**2))


def maximum_rotation_velocity(
    mass_dyn: Union[u.Quantity, np.ndarray, float],
    rnorm: Union[u.Quantity, np.ndarray, float],
) -> Union[u.Quantity, np.ndarray, float]:
    '''Maximum rotation velocity of Freeman disk function'''
    if isinstance(mass_dyn, u.Quantity):
        _mass_dyn = const.G * mass_dyn.to(u.kg)
    elif isinstance(mass_dyn, np.ndarray) or isinstance(mass_dyn, float):
        _mass_dyn = 10.0**mass_dyn

    if isinstance(rnorm, u.Quantity):
        _rnorm = rnorm.to(u.km)
    elif isinstance(rnorm, np.ndarray) or isinstance(rnorm, float):
        _rnorm = rnorm

    vmax = 0.88 * np.sqrt(_mass_dyn / (2 * _rnorm))
    if isinstance(vmax, u.Quantity):
        vmax = vmax.decompose().to(u.km / u.s)
    return vmax
