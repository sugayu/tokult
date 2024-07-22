'''Mock observation.

This class output mock-observed data cubes by combining galaxy models and
mock telescopes.
'''

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np


if TYPE_CHECKING:
    from ..parameters import ParameterManager
    from ..models import AbstractCubeBuilder
    from ..mocktelescope import MockTelescope


##
class MockObservation:
    '''Mock observation of a galaxy model using a mock telescope.'''

    def __init__(self) -> None:
        self.pmanager: ParameterManager
        self.models: AbstractCubeBuilder
        self.telescope: MockTelescope

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.be_conducted(p)

    def be_conducted(self, p: tuple[float, ...]) -> np.ndarray:
        '''Give model data cube generated from the input parameters.'''
        # TODO: How does it distribute parameters to models?
        fullparam = self.pmanager.restore(p)

        name_kin = self.models.galaxies._kinematic_model.modelname
        name_emi = self.models.galaxies._brightness_model.modelname
        name_cube = self.models.galaxies.modelname

        p_kin = self.pmanager.extract(fullparam, name_kin)
        p_emi = self.pmanager.extract(fullparam, name_emi)
        p_build = self.pmanager.extract(fullparam, name_cube)

        galaxy = self.models.build(p_kin, p_emi, p_build)

        # p_telescope = self.pmanager.extract(fullparam)
        cube = self.telescope.observe(galaxy)
        return cube
