'''Optimize cube models.
'''
from __future__ import annotations
from typing import Callable, Sequence, Optional, Union, TYPE_CHECKING
from dataclasses import dataclass
from collections.abc import ABCMeta, abstractmethod
from logging import getLogger

import numpy as np

from .fitparams import _FittedData, _ParameterConfig
from ..models import InputParams
from .. import misc

if TYPE_CHECKING:
    from .core import DataCube

logger = getLogger(__name__)
data: _FittedData
paramconfig: _ParameterConfig


##
class SolutionDI(ABCMeta):
    '''Absctract class of dependency injector for Solution.'''

    def __init__(self) -> None:
        ...


class Solution:
    '''Contains output solutions of fittings.'''

    def __init__(
        self,
        p_best: Union[list[float], tuple[float, ...]],
        error_high: Union[list[float], tuple[float, ...]],
        error_low: Union[list[float], tuple[float, ...]],
        chi2: float,
        dof: float,
        cov: np.ndarray,
        mode_fitting: str,
        optimizer: SolutionDI,
        # sampler: Optional[emcee.EnsembleSampler] = None,
        # params_mc: Optional[np.ndarray] = None,
        # output: Optional[OptimizeResult] = None,
    ) -> None:
        self.best = InputParams(*p_best)
        self.error_high = InputParams(*error_high)
        self.error_low = InputParams(*error_low)
        self.chi2 = chi2
        self.dof = dof
        self.cov = cov
        # self.sampler = sampler
        # self.params_mc = params_mc
        # self.output = output
        self.meta = self.MetaInfoOfSolution()

        global parameters_preset, index_free, index_fixp_target, index_fixp_source
        self.parameters_preset = np.copy(parameters_preset)
        self.index_free = np.copy(index_free)
        self.index_fixp_target = np.copy(index_fixp_target)
        self.index_fixp_source = np.copy(index_fixp_source)

    def set_metainfo(
        self, z: Optional[float] = None, header: Optional[fits.Header] = None
    ) -> None:
        '''Set meta infomation used in Solution'''
        keys_arguments = ['z', 'header']

        for k in keys_arguments:
            if (value_input := locals()[k]) is not None:
                setattr(self.meta, k, value_input)

    @dataclass
    class MetaInfoOfSolution:
        '''Meta data container of Solution.'''

        z: float = 0.0
        header: Optional[fits.Header] = None

    def add_units(
        self, params: Optional[Union[InputParams, np.ndarray]] = None
    ) -> FitParamsWithUnits:
        '''Get best parameters with physical units.'''
        if params is None:
            return self.best.to_units(header=self.meta.header, redshift=self.meta.z)
        elif isinstance(params, InputParams):
            return params.to_units(header=self.meta.header, redshift=self.meta.z)
        elif isinstance(params, np.ndarray):
            return InputParamsArray.from_ndarray(params).to_units(
                header=self.meta.header, redshift=self.meta.z
            )

    def restore_params(
        self, params: Union[np.ndarray, tuple[float, ...]]
    ) -> Union[np.ndarray, tuple[float, ...]]:
        '''Restore parameters with pfix by inserting parameters into params.'''
        if (parameters_preset is None) or (len(params) == 14):
            return params

        if isinstance(params, np.ndarray):
            if params.ndim == 2:
                newshape = (params.shape[0], 1)
                _parameters_preset = np.tile(self.parameters_preset, newshape).T
                _parameters_preset[self.index_free, :] = params
                _parameters_preset[self.index_fixp_target, :] = _parameters_preset[
                    self.index_fixp_source, :
                ]
                return _parameters_preset

        parameters_preset[self.index_free] = params
        parameters_preset[self.index_fixp_target] = parameters_preset[
            self.index_fixp_source
        ]
        return tuple(parameters_preset)


class Optimizer(ABCMeta):
    '''Abstract class to optimze cube models.'''

    def __init__(self) -> None:
        pass

    @abstractmethod
    def optimize(self) -> Solution:
        ...

    @abstractmethod
    def calculate_posterior(
        self,
        params: tuple[float, ...],
        model_func: Callable,
        bound: tuple[Sequence[float], Sequence[float]],
    ) -> float:
        '''Calcurate log probability.'''
        log_prior = self.calculate_log_prior(params, bound)
        if not np.isfinite(log_prior):
            return -np.inf
        chi = self.calculate_chi(params, model_func)
        log_likelihood = -0.5 * np.sum(abs(chi) ** 2)
        return log_prior + log_likelihood

    @abstractmethod
    def calculate_prior(
        self, params: tuple[float, ...], bound: tuple[Sequence[float], Sequence[float]]
    ) -> float:
        '''Calcurate log prior of parameters.'''
        _params = np.array(params)
        bound0, bound1 = (np.array(bound[0]), np.array(bound[1]))
        if np.all(bound0 < _params) and np.all(_params < bound1):
            return 0.0
        return -np.inf

    @abstractmethod
    def calculate_chi(
        self, params: tuple[float, ...], model_func: Callable
    ) -> np.ndarray:
        '''Calcurate chi = (data-model)/error.'''
        global cube, cube_error, mask
        model = model_func(params)
        if mask is not None:
            model = model[mask]
        chi = (cube - model) / cube_error
        return chi.ravel()


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
