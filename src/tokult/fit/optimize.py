'''Optimize cube models.
'''

from __future__ import annotations
from typing import Callable, Sequence, Optional, Union, TYPE_CHECKING
from abc import ABC, abstractmethod
from logging import getLogger

import numpy as np
from astropy.nddata import NDData

from .. import misc
from ..parameters import CompleteFittingParameters
from ..models import AbstractCubeBuilder
from ..mockobs import MockTelescope
from .fitparams import _FittedData, _ParameterConfig
from .solution import Solution

# if TYPE_CHECKING:
#     from .core import DataCube

logger = getLogger(__name__)
data: _FittedData
paramconfig: _ParameterConfig


##
class Optimizer(ABC):
    '''Abstract class to optimze cube models.'''

    def __init__(self) -> None:
        self.fullparams: CompleteFittingParameters
        self.model: AbstractCubeBuilder
        self.telescope: MockTelescope
        self.data: NDData

    @abstractmethod
    def optimize(self) -> Solution:
        '''Optimize fitting parameters to maximize the probability.'''
        ...

    @abstractmethod
    def calculate_probability(self, params: tuple[float, ...]) -> float:
        '''Calcurate log probability.'''
        ...

    @abstractmethod
    def calculate_prior(self, params: tuple[float, ...]) -> float:
        '''Calcurate log prior of parameters.'''
        ...

    @abstractmethod
    def calculate_likelihood(self, params: tuple[float, ...]) -> float:
        '''Calcurate chi = (data-model)/error.'''
        ...

    def modeling(self, p: tuple[float, ...]) -> np.ndarray:
        '''Give model data cube generated from the input parameters.'''
        p_kin, p_light, p_build = self.fullparams(p)
        galaxy = self.model.build(p_kin, p_light, p_build)
        cube = self.telescope.observe(galaxy)
        return cube


def initialize_data(
    datacube: DataCube,
    mask_for_fit: np.ndarray,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    noisescale_factor: float = 1.0,
    upsampling_rate: tuple[int, int, int] = (1, 1, 1),
) -> None:
    '''Construct _FittedData and set parameters.'''
    if not np.any(mask_for_fit):
        raise ValueError(
            '"mask_for_fit" filled by False. We believe that you don\'t want to use it.'
        )
    if (upsampling_rate[1] > 1) or (upsampling_rate[2] > 1):
        logger.warning(
            'Up-sampling in x and y has not yet been implemented. '
            'It performs fitting without up-sampling in x and y.'
        )

    global data
    cube = np.copy(datacube.imageplane)
    cube_error = datacube.rms()
    cube_error = cube_error[:, np.newaxis, np.newaxis]

    cube_error = np.broadcast_to(cube_error, cube.shape)
    cube = cube[mask]
    cube_error = cube_error[mask] * noisescale_factor
    vv_grid, yy_grid_image, xx_grid_image = datacube.coord_imageplane
    vv_grid = misc.gridding_upsample(vv_grid[:, 0, 0], upsampling_rate[0])
    vv_grid = vv_grid.reshape(-1, 1, 1)

    data = _FittedData(
        cube=cube,
        cube_error=cube_error,
        cubeshape=None,
        cubeshape_imageplane=cube.shape,
        xx_grid=xx_grid,
        yy_grid=yy_grid,
        vv_grid=vv_grid,
        xslice=datacube.xslice,
        yslice=datacube.yslice,
        lensing=func_lensing if func_lensing else misc.f_no_lensing,
        lensing_interpolation=(
            func_create_lensinginterp(xx_grid_image, yy_grid_image)
            if func_create_lensinginterp
            else misc.no_lensing_interpolation
        ),
        convolve=func_convolve if func_convolve else misc.f_no_convolve,
        mask=mask_for_fit,
    )


# CAUTION: Must unify initialization with this uv function
def initialize_globalparameters_for_uv(
    datacube: DataCube,
    beam_vis: np.ndarray,
    norm_weight: float,
    mask_for_fit: np.ndarray,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    noisescale_factor: float = 1.0,
    upsampling_rate: tuple[int, int, int] = (1, 1, 1),
) -> None:
    '''Set global parameters used in fitting.py in the uv plane.'''
    global cube, cube_error, cubeshape, cubeshape_imageplane
    global xx_grid, yy_grid, vv_grid, xslice, yslice
    global lensing, lensing_interpolation, mask

    size = datacube.original[0, :, :].size  # constant var needed for convolution
    cube = datacube.uvplane / beam_vis / size
    cube_error = np.sqrt(abs(beam_vis.real)) / beam_vis / np.sqrt(norm_weight) / size
    cube_error[cube_error == 0] = cube_error.max()  # to prevent divide-by-zero
    cube_error = _correct_cube_error_for_uv(cube_error)
    cubeshape = datacube.original[datacube.vslice, :, :].shape
    cubeshape_imageplane = datacube.imageplane.shape
    # sigma = (cube / cube_error).real
    # mask_to_remove_outlier = (sigma > -5) & (sigma < 5)
    # if mask_for_fit is not None:
    #     mask = mask_for_fit & mask_to_remove_outlier
    # else:
    #     mask = mask_to_remove_outlier

    vv_grid, yy_grid_image, xx_grid_image = datacube.coord_imageplane
    vv_grid = misc.gridding_upsample(vv_grid[:, 0, 0], upsampling_rate[0])
    vv_grid = vv_grid.reshape(-1, 1, 1)

    yy_grid = yy_grid_image[0][np.newaxis, :, :]
    xx_grid = xx_grid_image[0][np.newaxis, :, :]
    xx_grid, yy_grid = lensing(xx_grid, yy_grid)


def _correct_cube_error_for_uv(cube_error: np.ndarray) -> np.ndarray:
    '''Correct the cube error used in the uv-plane.

    On the uv-pane, the cube error is computed from the beam pattern.
    However, some pixels have sqrt(2) times larger errors and no imaginary parts
    becuase of characteristics of rfft, the Fourier transform of the real image.
    This function applys the correction of sqrt(2) to specific pixels.
    '''
    if cube_error.ndim == 3:
        _cube_error = np.copy(cube_error)
        shape = cube_error.shape
        idx_nyquist = shape[1] // 2
        i = ([0, 0, idx_nyquist, idx_nyquist], [0, -1, 0, -1])
        _cube_error[:, i[0], i[1]] *= np.sqrt(2)
        return _cube_error

    raise IndexError(f'Dimension of cube_error should be 3, but is {cube_error.ndim}.')


def _add_mask_for_uv(mask: np.ndarray) -> np.ndarray:
    '''Add the mask used for the uv-plane fitting.

    As similar to _correct_cube_error_for_uv, this function masks
    specific pixels that include redundant information because of
    characteristics of rfft.
    '''
    if mask.ndim == 3:
        shape = mask.shape
        idx_nyquist = shape[1] // 2
        mask[:, 1:idx_nyquist, [0, -1]] = False
        return mask

    raise IndexError(f'Dimension of mask should be 3, but is {mask.ndim}.')
