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
    '''Mock observation of a galaxy model using a mock telescope.'''

    def __init__(
        self,
        models: AbstractCubeBuilder = SimpleSkyCubeBuilder(),
        telescope=MockTelescope(),
    ) -> None:
        self.pmanager: ParameterManager
        self.models = models
        self.telescope = telescope

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.be_conducted(p)

    def be_conducted(self, p: tuple[float, ...]) -> np.ndarray:
        '''Give model data cube generated from the input parameters.'''
        fullparam = self.pmanager.restore(p)
        self.models.pmanager = self.pmanager
        galaxy = self.models.build(fullparam)

        # p_telescope = self.pmanager.extract(fullparam)
        cube = self.telescope.observe(galaxy)
        return cube
