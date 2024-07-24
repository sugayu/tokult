'''Build thin disk model cube.
'''

import numpy as np
import astropy.units as u

from ...utils import function
from ...parameters import FitPar, FittingParametersBase
from ..abstract import (
    AbstractGalaxyCube,
    AbstractBrightness,
    AbstractKinematics,
)

# for defaults
from ..kinematics import FreemanDiskRotation
from ..brightness import ExponentialProfile


##
class ThinDiskParameters(FittingParametersBase):
    sigma: FitPar = FitPar(unit=u.pix, bound=(0, np.inf), initial=1.0)


class ThinDisk(AbstractGalaxyCube):
    '''Build a thin-disk model cube.'''

    def __init__(
        self,
        kinematic_model: AbstractKinematics = FreemanDiskRotation(),
        brightness_model: AbstractBrightness = ExponentialProfile(),
    ) -> None:
        super().__init__(
            kinematic_model=kinematic_model,
            brightness_model=brightness_model,
        )
        self.coordinate_velocity: np.ndarray
        self.p = ThinDiskParameters()

    def output(
        self,
        p_kin: tuple[float, ...],
        p_light: tuple[float, ...],
        p_cube: tuple[float, ...],
    ) -> np.ndarray:
        velocity = self.kinematic_model.output(p_kin)
        brightness = self.brightness_model.output(p_light)
        sigma = p_cube[0]
        cube = function.gaussian(
            self.coord, center=velocity, sigma=sigma, area=brightness
        )
        return cube
