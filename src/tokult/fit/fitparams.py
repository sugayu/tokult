'''Global parameters used in fitting.

The main reason why these are treated as global parameters is a reduction of
data trafic in multiprocess.
'''
from typing import Callable, Optional
from dataclasses import dataclass
import numpy as np
from .. import misc


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
        misc.no_lensing
    )  # a function to convert source plane to image plane. Method of GravLens.
    lensing_interpolation: Callable = misc.no_lensing_interpolation
    convolve: Callable = (
        misc.no_convolve
    )  # a function to convolve the input datacube. Method of DirtyBeam.


@dataclass
class _ParameterConfig:
    '''Configuration to controle fitting parameters.'''

    parameters_preset: Optional[np.ndarray]
    index_free: list[int]
    index_fixp_target: list[int]
    index_fixp_source: list[int]
