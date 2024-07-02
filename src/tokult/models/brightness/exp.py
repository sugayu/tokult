'''Exponential brightness profile.
'''

# from typing import NamedTuple, TypeVar, Generic

import numpy as np
import astropy.units as u

from ...parameters import FittingParametersBase, FitPar
from ..abstract import AbstractBrightness
from ..utils import coordinates as coord


##
class ExponentialProfileParameters(FittingParametersBase):
    x0: FitPar = FitPar(unit=u.pix, bound=(-np.inf, np.inf), initial=0.0)
    y0: FitPar = FitPar(unit=u.pix, bound=(-np.inf, np.inf), initial=0.0)
    PA: FitPar = FitPar(unit=u.rad, bound=(0.0, 2 * np.pi), initial=3.0)
    inclination: FitPar = FitPar(unit=u.rad, bound=(0.0, np.pi / 2), initial=1.0)
    radius: FitPar = FitPar(unit=u.pix, bound=(0.0, np.inf), initial=1.0)
    brightness_center: FitPar = FitPar(
        unit=u.u.Jy / u.pix / u.pix, bound=(0.0, np.inf), initial=0.01
    )


class ExponentialProfile(AbstractBrightness):
    '''Exponential surface brightness profile.'''

    cls_param = ExponentialProfileParameters

    def __init__(self) -> None:
        self.coord: np.ndarray
        self.p = ExponentialProfileParameters()

    def output(self, _p: tuple[float, ...]) -> np.ndarray:
        p = self.p.namedtuplize(_p)
        coord_i = coord.to_relative_from(self.coord, at_x0=p.x0, at_y0=p.y0)
        rr_i, _ = coord.to_object_from(coord_i, PA=p.PA, incl=p.inclination)
        return reciprocal_exp(rr_i, norm=p.brightness_center, rnorm=p.radius)


def reciprocal_exp(r: np.ndarray, norm: float, rnorm: float) -> np.ndarray:
    '''Reciprocal exponential function'''
    return norm * np.exp(-r / rnorm)
