'''The most simple cube builder.
'''

import numpy as np
from ..abstract import AbstractCubeBuilder, AbstractGalaxyCube
from ..galaxy import ThinDisk


##
class SimpleCubeBuilder(AbstractCubeBuilder):
    '''The most simple cube builder.'''

    def __init__(self, galaxy_models: AbstractGalaxyCube = ThinDisk()) -> None:
        super().__init__(galaxy_models)

    def build(self, p) -> np.ndarray:
        return self.galaxies()
