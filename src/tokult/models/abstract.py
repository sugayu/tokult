'''Provide abstract classes, which will be inherited by all models.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..parameters import FittingParametersBase


##
class AbstractBrightness(ABC):
    '''Abstract class to give surface brightness profiles.'''

    def __init__(self, coordinate_abs: np.ndarray) -> None:
        self.coord = coordinate_abs
        self.p: FittingParametersBase

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.output(p)

    @abstractmethod
    def output(self, p: tuple[float, ...]) -> np.ndarray:
        '''Return surface brightness profiles (depending on the positions).'''
        pass


class AbstractKinematics(ABC):
    '''Abstract class to give kinematics.'''

    def __init__(self, coordinate_abs: np.ndarray) -> None:
        self.coord = coordinate_abs
        self.p: FittingParametersBase

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.output(p)

    @abstractmethod
    def output(self, p: tuple[float, ...]) -> np.ndarray:
        '''Return velocity profiles depending on the positions.'''
        pass


class AbstractCubeBuilder(ABC):
    '''Abstract class to build cube from brightness and kinematics models.'''

    def __init__(
        self,
        coordinate_velocity: np.ndarray,
        kinematic_model: AbstractKinematics,
        brightness_model: AbstractBrightness,
    ) -> None:
        self.coord = coordinate_velocity
        self._kinematic_model = kinematic_model
        self._brightness_model = brightness_model
        self.p: FittingParametersBase

    def __call__(
        self,
        p_kin: tuple[float, ...],
        p_brght: tuple[float, ...],
        p_bld: tuple[float, ...],
    ) -> np.ndarray:
        return self.build(p_kin, p_brght, p_bld)

    @abstractmethod
    def build(
        self,
        p_kin: tuple[float, ...],
        p_brght: tuple[float, ...],
        p_bld: tuple[float, ...],
    ) -> np.ndarray:
        '''Main method to build a model cube from full input parameters.'''
        ...
