'''Convert grids of a 3D cube.

This is a special layer in the mock telescope, which will be adopted before building the model, not after.
'''

from abc import abstractmethod
import numpy as np
from scipy.interpolate import RectBivariateSpline
from astropy import wcs
from .telescope import TelescopeLayer
from logging import getLogger

logger = getLogger(__name__)


##
class GridConverter(TelescopeLayer):
    '''Abstract class to convert data grids.

    If this is set in the mock observations, this class (function) should be passed when the grids are defined.
    '''

    def __init__(self) -> None:
        super().__init__()

    def __call__(self, grids: np.ndarray) -> np.ndarray:
        return self.convert(grids)

    @abstractmethod
    def convert(self, sky: np.ndarray) -> np.ndarray:
        '''Main method to convert the grid coordinate.'''
        pass
