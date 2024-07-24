'''Mock observation.

This class output mock-observed data cubes by combining galaxy models and
mock telescopes.
'''

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np


# for defaults
from ..models.builder import SimpleCubeBuilder
from ..mocktelescope import MockTelescope

if TYPE_CHECKING:
    from ..parameters import ParameterManager
    from ..models import AbstractCubeBuilder


##
class MockObservation:
    '''Mock observation of a galaxy model using a mock telescope.'''

    def __init__(
        self,
        models: AbstractCubeBuilder = SimpleCubeBuilder(),
        telescope=MockTelescope(),
    ) -> None:
        self.pmanager: ParameterManager
        self.models = models
        self.telescope = telescope

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.be_conducted(p)

    def be_conducted(self, p: tuple[float, ...]) -> np.ndarray:
        '''Give model data cube generated from the input parameters.'''
        # TODO: How does it distribute parameters to models?
        fullparam = self.pmanager.restore(p)

        name_kin = self.models.galaxies.kinematic_model.name
        name_emi = self.models.galaxies.brightness_model.name
        name_cube = self.models.galaxies.name

        p_kin = self.pmanager.extract(fullparam, name_kin)
        p_emi = self.pmanager.extract(fullparam, name_emi)
        p_build = self.pmanager.extract(fullparam, name_cube)

        galaxy = self.models.build(p_kin, p_emi, p_build)

        # p_telescope = self.pmanager.extract(fullparam)
        cube = self.telescope.observe(galaxy)
        return cube
