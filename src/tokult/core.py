'''Core of Tokult to connect complex modules to achive the goal.

This module treats:
- Passing objects
- Running optimization
- Executing parallelization
'''

from dataclasses import dataclass
from astropy.nddata import NDData

from .fit import Optimizer, Solution
from .models import AbstractCubeBuilder
from .mockobs import MockTelescope

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
        optimizer: Optimizer | None,
        models: AbstractCubeBuilder | None,
        telescope: MockTelescope | None,
    ) -> None:
        default = Defaults()
        self.data = default.data if data is None else data
        self.optimizer = default.optimizer if optimizer is None else optimizer
        self.models = default.models if models is None else models
        self.telescope = default.telescope if telescope is None else telescope

    def runfit(self) -> Solution:
        sol = self.optimizer.optimize()
        return sol


@dataclass
class Defaults:
    data: NDData = NDData([])
    optimizer: Optimizer = EmceeMCMC()
    models: AbstractCubeBuilder = ThinDiskBuilder()
    telescope: MockTelescope = MockTelescope()
