'''Build thin disk model cube.
'''

import numpy as np
import astropy.units as u

from ... import function
from ...parameters import FitPar, FittingParametersBase
from ..abstract import (
    AbstractCubeBuilder,
    AbstractBrightness,
    AbstractKinematics,
)


##
class ThinDiskParameters(FittingParametersBase):
    sigma: FitPar = FitPar(unit=u.pix, bound=(0, np.inf), initial=1.0)


class ThinDiskBuilder(AbstractCubeBuilder):
    '''Build a thin-disk model cube.'''

    def __init__(
        self,
        coordinate_velocity: np.ndarray,
        kinematic_model: AbstractKinematics,
        brightness_model: AbstractBrightness,
    ) -> None:
        super().__init__(
            coordinate_velocity=coordinate_velocity,
            kinematic_model=kinematic_model,
            brightness_model=brightness_model,
        )
        self.p = ThinDiskParameters()

    def build(
        self,
        p_kin: tuple[float, ...],
        p_brght: tuple[float, ...],
        p_bld: tuple[float, ...],
    ) -> np.ndarray:
        velocity = self._kinematic_model.output(p_kin)
        brightness = self._brightness_model.output(p_brght)
        sigma = p_bld[0]
        model = function.gaussian(
            self.coord, center=velocity, sigma=sigma, area=brightness
        )
        return model
