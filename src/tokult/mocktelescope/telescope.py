'''Mock observation pipeline.
'''

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    from ..parameters import FittingParametersBase

__all__ = ['MockTelescope', 'TelescopeLayer']


##
class MockTelescope:
    '''Telescope-like class to provide mock observations.'''

    def __init__(self) -> None:
        self.layers: list[TelescopeLayer] = []

    def observe(self, skymodel: np.ndarray) -> np.ndarray:
        '''Mock observation of the skymodel (image or cube).'''
        if not self.layers:
            return skymodel

        _skymodel = skymodel.copy()
        for layer in self.layers:
            _skymodel = layer(_skymodel)
        return _skymodel


class TelescopeLayer(ABC):
    '''Layer that transform model cubes during observations.'''

    def __init__(self) -> None:
        self.p: FittingParametersBase | None

    @abstractmethod
    def __call__(self, sky: np.ndarray) -> np.ndarray: ...
