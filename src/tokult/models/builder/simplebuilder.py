'''The most simple cube builder.
'''

import numpy as np
from ..abstract import AbstractCubeBuilder, AbstractGalaxyCube
from ..galaxy import ThinDisk


##
class SimpleSkyCubeBuilder(AbstractCubeBuilder):
    '''The most simple cube builder.'''

    def __init__(self, galaxy_models: AbstractGalaxyCube = ThinDisk()) -> None:
        super().__init__(galaxy_models)

    def build(self, p) -> np.ndarray:
        # TODO: How does it distribute parameters to models?
        name_kin = self.galaxies.kinematic_model.name
        name_emi = self.galaxies.brightness_model.name
        name_cube = self.galaxies.name

        p_kin = self.pmanager.extract(p, name_kin)
        p_emi = self.pmanager.extract(p, name_emi)
        p_build = self.pmanager.extract(p, name_cube)
        return self.galaxies(p_kin, p_emi, p_build)
