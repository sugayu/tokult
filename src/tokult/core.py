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
from .mockobs import MockTelescope
from .parameters import ParameterManager

# for defaults
from .fit.algorithms import EmceeMCMC
from .models.builder import ThinDiskBuilder

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
        optimizer: Optimizer | None,
    ) -> None:
        default = Defaults()
        self.data = default.data if data is None else data
        self.models = default.models if models is None else models
        self.telescope = default.telescope if telescope is None else telescope
        self.optimizer = default.optimizer if optimizer is None else optimizer
        self.fullparams: ParameterManager

    def runfit(self) -> Solution:
        self.ready_fittingparameters()

        self.optimizer.data = self.data
        self.optimizer.models = self.models
        self.optimizer.telescope = self.telescope
        self.optimizer.fullparams = self.fullparams

        sol = self.optimizer.optimize()
        return sol

    def ready_fittingparameters(self) -> ParameterManager:
        return ParameterManager()


@dataclass
class Defaults:
    data: NDData = NDData([])
    optimizer: Optimizer = EmceeMCMC()
    models: AbstractCubeBuilder = ThinDiskBuilder()
    telescope: MockTelescope = MockTelescope()
