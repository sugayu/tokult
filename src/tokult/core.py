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
from astropy.nddata import NDData

from .fit import Optimizer, Solution
from .models import AbstractCubeBuilder
from .mocktelescope import MockTelescope
from .mockobs import MockObservation
from .parameters import ParameterManager

# for defaults
from .fit.algorithms import EmceeMCMC
from .models.builder import SimpleCubeBuilder

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

    def runfit(self) -> Solution:
        self.pmanager = self.standby_fittingparameters()

        self.observation.models = self.models
        self.observation.telescope = self.telescope
        self.optimizer.observation = self.observation
        self.optimizer.data = self.data
        self.optimizer.pmanager = self.pmanager

        sol = self.optimizer.optimize()
        return sol

    def standby_fittingparameters(self) -> ParameterManager:
        return ParameterManager(mockobs=self.observation, optimizer=self.optimizer)


@dataclass
class Defaults:
    data: NDData = NDData([])
    optimizer: Optimizer = EmceeMCMC()
    models: AbstractCubeBuilder = SimpleCubeBuilder()
    observation: MockObservation = MockObservation()
    telescope: MockTelescope = MockTelescope()
