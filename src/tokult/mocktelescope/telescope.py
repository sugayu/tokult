'''Mock observation pipeline.
'''

from typing import TYPE_CHECKING
from abc import ABC, abstractmethod
import numpy as np

if TYPE_CHECKING:
    from ..parameters import FittingParametersBase

__all__ = ['MockTelescope']


##
class MockTelescope:
    '''Telescope-like class to provide mock observations.'''

    def __init__(self) -> None:
        self.components: list[TelescopeComponent] = []

    def observe(self, obj: np.ndarray) -> np.ndarray:
        '''Mock observation of the model object (image or cube).'''
        return obj


class TelescopeComponent(ABC):
    '''Components that transform model cubes during observations.'''

    def __init__(self) -> None:
        self.p: FittingParametersBase | None
