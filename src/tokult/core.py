'''Core of Tokult to connect complex modules to achive the goal.

This module treats:
- Passing objects
- Running optimization
- Executing parallelization

Users can pass data, models, and all necessary information in any order they like.
This Core class treats these objects in proper way and passes them to the next classes,
including Optimizer.
'''

from dataclasses import dataclass
import numpy as np
from astropy.nddata import NDData

from .fit import Optimizer, Solution
from .models import AbstractCubeBuilder
from .mocktelescope import MockTelescope
from .mockobs import MockObservation
from .parameters import ParameterManager

# for defaults
from .fit.algorithms import EmceeMCMC
from .models.builder import SimpleSkyCubeBuilder

__all__ = ['Core']


##
class Core:
    '''Core manipulations of fitting.

    This class has a resoponsivility to pass necesary objects to fitting interfaces.
    '''

    def __init__(
        self,
        data: NDData | None,
        models: AbstractCubeBuilder | None,
        telescope: MockTelescope | None,
        observation: MockObservation | None,
        optimizer: Optimizer | None,
    ) -> None:
        default = Defaults()
        self.data = default.data if data is None else data
        self.models = default.models if models is None else models
        self.telescope = default.telescope if telescope is None else telescope
        self.observation = default.observation if observation is None else observation
        self.optimizer = default.optimizer if optimizer is None else optimizer
        self.pmanager: ParameterManager

    def runfit(self, initial=np.ndarray | None) -> Solution:
        self.pmanager = self.standby_fittingparameters()

        assert len(self.data.data.shape) == 3
        coord_yx, coord_v = self.get_3Dpositiongrids()
        self.observation.models.coord_yx = coord_yx
        self.observation.models.coord_velocity = coord_v

        self.observation.pmanager = self.pmanager
        self.optimizer.pmanager = self.pmanager

        self.observation.models = self.models
        self.observation.telescope = self.telescope
        self.optimizer.observation = self.observation
        self.optimizer.data = self.data

        sol = self.optimizer.optimize(initial=initial)
        return sol

    def build_model(self, p: tuple[float]) -> np.ndarray:
        self.pmanager = self.standby_fittingparameters()
        coord_yx, coord_v = self.get_3Dpositiongrids()
        self.observation.models.coord_yx = coord_yx
        self.observation.models.coord_velocity = coord_v

        self.observation.pmanager = self.pmanager

        self.observation.models = self.models
        self.observation.telescope = self.telescope
        return self.observation(p)

    def standby_fittingparameters(self) -> ParameterManager:
        return ParameterManager(mockobs=self.observation, optimizer=self.optimizer)

    def get_3Dpositiongrids(self) -> tuple[np.ndarray, np.ndarray]:
        '''Get 3D positional coordinate grids.'''
        nv, ny, nx = self.data.data.shape
        coord_yx = np.array(np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij'))
        coord_yx = np.moveaxis(coord_yx, 0, -1)
        coord_v = np.arange(nv).reshape((nv, 1, 1))
        return coord_yx, coord_v


@dataclass
class Defaults:
    data: NDData = NDData([])
    optimizer: Optimizer = EmceeMCMC()
    models: AbstractCubeBuilder = SimpleSkyCubeBuilder()
    observation: MockObservation = MockObservation()
    telescope: MockTelescope = MockTelescope()
