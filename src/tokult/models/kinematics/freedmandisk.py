'''Rotational kinematics of Freeman disk.
'''

# from typing import NamedTuple, TypeVar, Generic

import numpy as np
import scipy.special as sps
import astropy.units as u

from ...parameters import FittingParametersBase, FitPar
from ..abstract import AbstractKinematics
from ..utils import coordinates as coord


##
class FreemanDiskParameters(FittingParametersBase):
    x0: FitPar = FitPar(unit=u.pix, bound=(-np.inf, np.inf), initial=0.0)
    y0: FitPar = FitPar(unit=u.pix, bound=(-np.inf, np.inf), initial=0.0)
    PA: FitPar = FitPar(unit=u.rad, bound=(0.0, 2 * np.pi), initial=3.0)
    inclination: FitPar = FitPar(unit=u.rad, bound=(0.0, np.pi / 2), initial=1.0)
    radius: FitPar = FitPar(unit=u.pix, bound=(0.0, np.inf), initial=1.0)
    velocity_sys: FitPar = FitPar(unit=u.pix, bound=(-np.inf, np.inf), initial=1.0)
    mass_dyn: FitPar = FitPar(
        unit=u.dex(u.pix**3), bound=(-np.inf, np.inf), initial=1.0
    )


class FreemanDiskRotation(AbstractKinematics):
    '''Kinetic profile of Freeman disk'''

    def __init__(self) -> None:
        self.coord: np.ndarray
        self.p = FreemanDiskParameters()

    def output(self, _p: tuple[float, ...]) -> np.ndarray:
        p = self.p.namedtuplize(_p)
        coord_v = coord.to_relative_from(self.coord, at_x0=p.x0, at_y0=p.y0)
        rr, pphi = coord.to_object_from(coord_v, PA=p.PA, incl=p.inclination)
        velocity = p.velocity_sys + freemandisk(
            rr, pphi, mass_dyn=10.0**p.mass_dyn, rnorm=p.radius, incl=p.inclination
        )
        return velocity


def freemandisk(
    r: np.ndarray, phi: np.ndarray, mass_dyn: float, rnorm: float, incl: float
) -> np.ndarray:
    '''Freeman disk function'''
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
