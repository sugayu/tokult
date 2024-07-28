'''Global parameters used in fitting.

The main reason why these are treated as global parameters is a reduction of
data trafic in multiprocess.
'''

from typing import Callable
from dataclasses import dataclass
import numpy as np
from ..utils import dummy


##
@dataclass
class _FittedData:
    '''Container of input data that is necessary for fitting.'''

    cube: np.ndarray  # input DataCube contaning images.
    cube_error: np.ndarray
    cubeshape: tuple[int, ...]
    cubeshape_imageplane: tuple[int, ...]
    mask: np.ndarray  # mask_FoV: np.ndarray
    xx_grid: np.ndarray  # coordinate indices of x for the image datacube.
    yy_grid: np.ndarray  # coordinate indices of y for the image datacube.
    vv_grid: np.ndarray  # coordinate indices of v for the image datacube.
    xslice: slice
    yslice: slice
    lensing: Callable = (
        dummy.no_lensing
    )  # a function to convert source plane to image plane. Method of GravLens.
    lensing_interpolation: Callable = dummy.no_lensing_interpolation
    convolve: Callable = (
        dummy.no_convolve
    )  # a function to convolve the input datacube. Method of DirtyBeam.
