'''Mock observation.

This class output mock-observed data cubes by combining galaxy models and
mock telescopes.
'''

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np


# for defaults
from ..models.builder import SimpleSkyCubeBuilder
from ..mocktelescope import MockTelescope

if TYPE_CHECKING:
    from ..parameters import ParameterManager
    from ..models import AbstractCubeBuilder


##
class MockObservation:
    '''Mock observation of a galaxy model using a mock telescope.

    Responsibility:
        - Recieve a fitting parameter set from users and Optimizer.
        - Build a sky cube and then observe the sky to obtain the mock data cube.
        - Retrun the observed cube data.
    '''

    def __init__(
        self,
        models: AbstractCubeBuilder | None = None,
        telescope: MockTelescope | None = None,
    ) -> None:
        if models is None:
            models = SimpleSkyCubeBuilder()
        if telescope is None:
            telescope = MockTelescope()
        self.pmanager: ParameterManager
        self.models = models
        self.telescope = telescope

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.run(p)

    def run(self, p: tuple[float, ...]) -> np.ndarray:
        '''Give model data cube generated from the input parameters.'''
        fullparam = self.pmanager.restore(self.pmanager.convert(p))
        self.models.pmanager = self.pmanager
        cube_sky = self.models.build(fullparam)

        # p_telescope = self.pmanager.extract(fullparam)
        cube_obs = self.telescope.observe(cube_sky)
        return cube_obs
