'''Provide abstract classes, which will be inherited by all models.
'''

from __future__ import annotations
from abc import ABC, abstractmethod
import numpy as np
from ..parameters import FittingParametersBase


__all__ = [
    'AbstractBrightness',
    'AbstractKinematics',
    'AbstractGalaxyCube',
    'AbstractCubeBuilder',
]


##
class AbstractBrightness(ABC):
    '''Abstract class to give surface brightness profiles.'''

    def __init__(self) -> None:
        self.coord: np.ndarray
        self.p: FittingParametersBase

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.output(p)

    @abstractmethod
    def output(self, p: tuple[float, ...]) -> np.ndarray:
        '''Return surface brightness profiles (depending on the positions).'''
        pass

    @property
    def name(self) -> str:
        return self.p.name

    @name.setter
    def name(self, value: str) -> None:
        self.p.name = value


class AbstractKinematics(ABC):
    '''Abstract class to give kinematics.'''

    def __init__(self) -> None:
        self.coord: np.ndarray
        self.p: FittingParametersBase

    def __call__(self, p: tuple[float, ...]) -> np.ndarray:
        return self.output(p)

    @abstractmethod
    def output(self, p: tuple[float, ...]) -> np.ndarray:
        '''Return velocity profiles depending on the positions.'''
        pass

    @property
    def name(self) -> str:
        return self.p.name

    @name.setter
    def name(self, value: str) -> None:
        self.p.name = value


class AbstractGalaxyCube(ABC):
    '''Abstract class to build a galaxy cube from brightness and kinematics models.'''

    def __init__(
        self,
        kinematic_model: AbstractKinematics,
        brightness_model: AbstractBrightness,
    ) -> None:
        self.coord: np.ndarray
        self.p: FittingParametersBase
        self.kinematic_model = kinematic_model
        self.brightness_model = brightness_model

    def __call__(
        self,
        p_kin: tuple[float, ...],
        p_light: tuple[float, ...],
        p_build: tuple[float, ...],
    ) -> np.ndarray:
        return self.output(p_kin, p_light, p_build)

    @abstractmethod
    def output(
        self,
        p_kin: tuple[float, ...],
        p_light: tuple[float, ...],
        p_cube: tuple[float, ...],
    ) -> np.ndarray:
        '''Main method to build a model cube from full input parameters.'''
        ...

    @property
    def name(self) -> str:
        return self.p.name

    @name.setter
    def name(self, value: str) -> None:
        self.p.name = value


class AbstractCubeBuilder(ABC):
    '''Abstract class to build cube by combining galaxy cubes.'''

    def __init__(
        self,
        galaxy_models: AbstractGalaxyCube,
    ) -> None:
        self.coord: np.ndarray
        self.p: FittingParametersBase
        self.galaxies = galaxy_models

    def __call__(
        self,
        p_kin: tuple[float, ...],
        p_light: tuple[float, ...],
        p_build: tuple[float, ...],
    ) -> np.ndarray:
        return self.build(p_kin, p_light, p_build)

    @abstractmethod
    def build(
        self,
        p_kin: tuple[float, ...],
        p_light: tuple[float, ...],
        p_build: tuple[float, ...],
    ) -> np.ndarray:
        '''Main method to build a model cube from full input parameters.'''
        ...

    @property
    def name(self) -> str:
        return self.p.name

    @name.setter
    def name(self, value: str) -> None:
        self.p.name = value
