'''Modules of fitting functions
'''
from __future__ import annotations
from dataclasses import dataclass, field

# import time
# import pickle
# from pathlib import Path
import numpy as np
from numpy.random import default_rng
from scipy.optimize import least_squares as sp_least_squares
from scipy.optimize.optimize import OptimizeResult
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import tqdm
from typing import Callable, Sequence, Optional, Union, TYPE_CHECKING
from typing import NamedTuple
import emcee
from multiprocessing.pool import Pool
from . import function as func
from . import misc
from . import common as c

if TYPE_CHECKING:
    from .core import DataCube

__all__ = ['InputParams', 'FitParamsWithUnits', 'get_bound_params']

##
'''Global parameters used for fitting

data: input DataCube contaning images.
xx_grid: coordinate indices of x for the image datacube.
yy_grid: coordinate indices of y for the image datacube.
vv_grid: coordinate indices of v for the image datacube.
convolve: a function to convolve the input datacube. Method of DirtyBeam.
lensing: a function to convert source plane to image plane. Method of GravLens.
'''

cube: np.ndarray
cube_error: np.ndarray
cubeshape: tuple[int, ...]
xx_grid: np.ndarray
yy_grid: np.ndarray
vv_grid: np.ndarray
xslice: slice
yslice: slice
lensing: Callable = misc.no_lensing
lensing_interpolation: Callable = misc.no_lensing_interpolation
convolve: Callable = misc.no_convolve
mask: np.ndarray
# mask_FoV: np.ndarray

# To fix parameters in fitting
parameters_preset: Optional[np.ndarray]
index_free: list[int]
index_fixp_target: list[int]
index_fixp_source: list[int]


def least_square(
    datacube: DataCube,
    init: Sequence[float],
    bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    fix: Optional[FixParams] = None,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    beam_vis: Optional[np.ndarray] = None,
    norm_weight: Optional[float] = None,
    mask_for_fit: Optional[np.ndarray] = None,
    niter: int = 1,
    mode_fit: str = 'image',
    is_separate: bool = False,
) -> Solution:
    '''Least square fitting using scipy.optimize.least_squares
    '''
    if mask_for_fit is None:
        mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)
    if mode_fit == 'image':
        initialize_globalparameters_for_image(
            datacube,
            mask_for_fit,
            func_convolve,
            func_lensing,
            func_create_lensinginterp,
        )
        func_fit = construct_convolvedmodel
    elif mode_fit == 'uv':
        if beam_vis is None:
            raise ValueError('"beam_vis" is necessary for uvfit')
        if norm_weight is None:
            raise ValueError('Param "norm_weight" is necessary for uvfit.')
        if mask_for_fit is None:
            mask_for_fit = np.ones_like(datacube.uvplane).astype(bool)

        initialize_globalparameters_for_uv(
            datacube,
            beam_vis,
            norm_weight,
            mask_for_fit,
            func_lensing,
            func_create_lensinginterp,
        )
        func_fit = construct_uvmodel
    else:
        raise ValueError(f'mode_fit is "image" or "uv", no option for "{mode_fit}".')

    set_fixedparameters(fix, is_separate)
    bound = get_bound_params() if bound is None else bound
    _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
    if is_init_outside_of_bound(_init, _bound):
        raise ValueError('The "init" is outside of the "bound".')
    args = (func_fit,)

    for _ in range(niter):
        output = sp_least_squares(calculate_chi, _init, args=args, bounds=_bound)
        _init = output.x

    dof = datacube.imageplane.size - 1 - len(_init)
    chi2 = np.sum(calculate_chi(output.x, func_fit) ** 2.0)
    return Solution.from_leastsquare(output, chi2, dof)


def montecarlo(
    datacube: DataCube,
    init: Sequence[float],
    bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    fix: Optional[FixParams] = None,
    func_convolve: Optional[Callable] = None,
    func_fullconvolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    mask_for_fit: Optional[np.ndarray] = None,
    nperturb: int = 1000,
    niter: int = 1,
    is_separate: bool = False,
    progressbar: bool = False,
) -> Solution:
    '''Monte Carlo fitting to derive errors using scipy.optimize.least_squares
    '''
    if mask_for_fit is None:
        mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)
    initialize_globalparameters_for_image(
        datacube, mask_for_fit, func_convolve, func_lensing, func_create_lensinginterp
    )
    func_fit = construct_convolvedmodel

    set_fixedparameters(fix, is_separate)
    bound = get_bound_params() if bound is None else bound
    _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
    if is_init_outside_of_bound(_init, _bound):
        raise ValueError('The "init" is outside of the "bound".')
    args = (func_fit,)

    params_mc = np.empty((nperturb, len(_init)))
    rms_of_standardnoise = datacube._estimate_rms_of_standardnoise(
        shape=datacube.original.shape, convolve=func_fullconvolve
    )
    for j in tqdm.tqdm(range(nperturb), leave=None, disable=(not progressbar)):
        _init_j = _init
        global cube, mask
        cube = datacube.perturbed(
            convolve=func_fullconvolve, rms_of_standardnoise=rms_of_standardnoise
        )[mask]
        for _ in range(niter):
            output = sp_least_squares(calculate_chi, _init_j, args=args, bounds=_bound)
            _init_j = output.x
        params_mc[j, :] = output.x

    dof = datacube.imageplane.size - 1 - len(_init)
    chi2 = np.sum(calculate_chi(output.x, func_fit) ** 2.0)
    return Solution.from_montecarlo(params_mc, chi2, dof)


def mcmc(
    config: c.ConfigParameters,
    datacube: DataCube,
    init: Sequence[float],
    bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    fix: Optional[FixParams] = None,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    beam_vis: Optional[np.ndarray] = None,
    norm_weight: Optional[float] = None,
    mask_for_fit: Optional[np.ndarray] = None,
    mode_fit: str = 'image',
    is_separate: bool = False,
    nwalkers: int = 64,
    nsteps: int = 5000,
    pool: Optional[Pool] = None,
    progressbar: bool = False,
) -> Solution:
    '''MCMC using emcee
    '''
    rng = default_rng(222)

    if mode_fit == 'image':
        if mask_for_fit is None:
            mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)
        initialize_globalparameters_for_image(
            datacube,
            mask_for_fit,
            func_convolve,
            func_lensing,
            func_create_lensinginterp,
        )
        func_fit = construct_convolvedmodel
    elif mode_fit == 'uv':
        if beam_vis is None:
            raise ValueError('Parameter "beam_vis" is necessary for uvfit.')
        if norm_weight is None:
            raise ValueError('Parameter "norm_weight" is necessary for uvfit.')
        if mask_for_fit is None:
            mask_for_fit = np.ones_like(datacube.uvplane).astype(bool)

        initialize_globalparameters_for_uv(
            datacube,
            beam_vis,
            norm_weight,
            mask_for_fit,
            func_lensing,
            func_create_lensinginterp,
        )
        func_fit = construct_uvmodel
    else:
        raise ValueError(f'mode_fit is "image" or "uv", no option for "{mode_fit}".')

    set_fixedparameters(fix, is_separate)
    bound = get_bound_params() if bound is None else bound
    _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
    if is_init_outside_of_bound(_init, _bound):
        raise ValueError('The "init" is outside of the "bound".')
    args = (func_fit, _bound)

    ndim = len(_init)
    __init = np.array(_init)
    norm = rng.standard_normal((nwalkers, ndim))
    __init = __init + __init * config.mcmc_init_dispersion * norm

    if pool is not None:
        map_globals_to_childprocesses(pool)

    sampler = emcee.EnsembleSampler(
        nwalkers,
        ndim,
        calculate_log_probability,
        args=args,
        pool=pool,
        moves=config.mcmc_moves,
    )
    sampler.run_mcmc(__init, nsteps, progress=progressbar)

    dof = datacube.imageplane.size - 1 - len(_init)
    return Solution.from_sampler(sampler, dof, func_fit)


def calculate_chi(params: tuple[float, ...], model_func: Callable,) -> np.ndarray:
    '''Calcurate chi = (data-model)/error for least square fitting
    '''
    global cube, cube_error, mask
    model = model_func(params)
    if mask is not None:
        model = model[mask]

        # # Choose fitting region via indices.
        # iv = (v - dv * c.conf.chan_start - vel_start) / dv
        # iv = np.round(iv).astype(int)
        # ix = (x - xlow).astype(int)
        # iy = (y - ylow).astype(int)
        # model_last = model[iy, ix, iv]

    chi = (cube - model) / cube_error
    return chi.ravel()


def calculate_log_probability(
    params: tuple[float, ...],
    model_func: Callable,
    bound: tuple[Sequence[float], Sequence[float]],
) -> float:
    '''Calcurate log probability for MCMC technique.
    '''
    log_prior = calculate_log_prior(params, bound)
    if not np.isfinite(log_prior):
        return -np.inf
    chi = calculate_chi(params, model_func)
    log_likelihood = -0.5 * np.sum(abs(chi) ** 2)
    return log_prior + log_likelihood


def calculate_log_prior(
    params: tuple[float, ...], bound: tuple[Sequence[float], Sequence[float]]
) -> float:
    '''Calcurate log prior of parameters for MCMC technique.
    '''
    _params = np.array(params)
    bound0, bound1 = (np.array(bound[0]), np.array(bound[1]))
    if np.all(bound0 < _params) and np.all(_params < bound1):
        return 0.0
    return -np.inf


def construct_convolvedmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    global convolve
    model = construct_model_at_imageplane(params)
    model_convolved = convolve(model)
    # model_convolved = np.empty_like(model)
    # for i, image in enumerate(model):
    #     model_convolved[i, :, :] = convolve(image, index=i)
    return model_convolved


def construct_uvmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    global cubeshape, yslice, xslice
    model_cutout = construct_model_at_imageplane(params)
    model_image = np.zeros(cubeshape)
    model_image[:, yslice, xslice] = model_cutout
    # model_image = construct_model_at_imageplane(params)
    model_visibility = misc.rfft2(model_image)
    # image = misc.ifft2(model_visibility * beam_visibility)
    # return misc.fft2(image * mask_FoV)
    # return model_visibility * beam_visibility
    return model_visibility


def construct_model_at_imageplane(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube on image plane using parameters and the grav. lensing.
    '''
    global xx_grid, yy_grid, vv_grid
    _params = restore_params(params)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = _params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    # velocity field
    coord_v = to_relativecoord_from(coordinate_abs, at_x0=p0, at_y0=p1)
    rr, pphi = to_objectcoord_from(coord_v, PA=p2, incl=p3)
    velocity = p5 + func.freeman_disk(rr, pphi, mass_dyn=10.0 ** p6, rnorm=p4, incl=p3)

    # spatial intensity distribution
    coord_i = to_relativecoord_from(coordinate_abs, at_x0=p10, at_y0=p11)
    rr_i, _ = to_objectcoord_from(coord_i, PA=p12, incl=p13)
    intensity = func.reciprocal_exp(rr_i, norm=p7, rnorm=p9)

    # create cube
    model = func.gaussian(vv_grid, center=velocity, sigma=p8, area=intensity)
    return model


def to_objectcoord_from(
    coord_celestial: np.ndarray, PA: float, incl: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Convert coordinates from celestial coordinates to object polar coordinates.
    '''
    pa = PA
    inclination = incl

    # coord_source = lensing(coord_image)
    coord_object = misc.rotate_coord(coord_celestial, pa)
    xx, yy = np.moveaxis(coord_object, -1, 0)
    yy = yy / np.cos(inclination)
    r, phi = misc.polar_coord(xx, yy)
    return r, phi


def to_relativecoord_from(
    coord_source: np.ndarray, at_x0: float, at_y0: float
) -> np.ndarray:
    '''Convert coordinates from absolute positions to relative positions.
    '''
    global lensing_interpolation
    central_position = lensing_interpolation(at_x0, at_y0)
    return coord_source - central_position[np.newaxis, np.newaxis, :]


def construct_model_at_imageplane_with(
    params: tuple[float, ...],
    xx_grid_image: np.ndarray,
    yy_grid_image: np.ndarray,
    vv_grid_image: np.ndarray,
    lensing: Optional[Callable] = None,
    create_interpolate_lensing: Optional[Callable] = None,
) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    lensing = lensing if lensing else misc.no_lensing
    xx_grid, yy_grid = lensing(xx_grid_image[[0], :, :], yy_grid_image[[0], :, :])
    vv_grid = vv_grid_image
    lensing_interpolation = (
        create_interpolate_lensing(xx_grid_image, yy_grid_image)
        if create_interpolate_lensing
        else misc.no_lensing_interpolation
    )

    keys_globals = ['xx_grid', 'yy_grid', 'vv_grid', 'lensing', 'lensing_interpolation']
    _globals = {}

    for k in keys_globals:
        try:
            _globals[k] = globals()[k]
        except KeyError:
            _globals[k] = None
        globals()[k] = locals()[k]

    model = construct_model_at_imageplane(params)

    for k in keys_globals:
        if _globals[k] is not None:
            globals()[k] = _globals[k]
        else:
            del globals()[k]

    return model


def construct_model_moment0(params: list[float]) -> np.ndarray:
    '''Construct a model moment0 map convolved with dirtybeam.
    '''
    global xx_grid, yy_grid
    p0, p1, p2, p3, p4, p5 = params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    coord_i = to_relativecoord_from(coordinate_abs, at_x0=p0, at_y0=p1)
    rr_i, _ = to_objectcoord_from(coord_i, PA=p2, incl=p3)
    intensity = func.reciprocal_exp(rr_i, norm=p5, rnorm=p4)
    model = convolve(intensity)

    return model


def construct_model_moment1(params: list[float]) -> np.ndarray:
    '''Construct a model moment1 map convolved with dirtybeam.
    '''
    global xx_grid, yy_grid
    p0, p1, p2, p3, p4, p5, p6 = params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    coord_v = to_relativecoord_from(coordinate_abs, at_x0=p5, at_y0=p6)
    rr, pphi = to_objectcoord_from(coord_v, PA=p4, incl=p0)
    velocity = p3 + func.freeman_disk(rr, pphi, mass_dyn=10.0 ** p2, rnorm=p1, incl=p0)
    # # NOTE: convolving velocity is correct?
    # model = convolve(velocity, index=0)

    return velocity


class Solution:
    '''Contains output solutions of fittings.
    '''

    def __init__(
        self,
        p_best: Union[list[float], tuple[float, ...]],
        error_high: Union[list[float], tuple[float, ...]],
        error_low: Union[list[float], tuple[float, ...]],
        chi2: float,
        dof: float,
        cov: np.ndarray,
        mode_fitting: str,
        sampler: Optional[emcee.EnsembleSampler] = None,
        params_mc: Optional[np.ndarray] = None,
        output: Optional[OptimizeResult] = None,
    ) -> None:
        self.best = InputParams(*p_best)
        self.error_high = InputParams(*error_high)
        self.error_low = InputParams(*error_low)
        self.chi2 = chi2
        self.dof = dof
        self.cov = cov
        self.sampler = sampler
        self.params_mc = params_mc
        self.output = output
        self.meta = self.MetaInfoOfSolution()

        global parameters_preset, index_free, index_fixp_target, index_fixp_source
        self.parameters_preset = np.copy(parameters_preset)
        self.index_free = np.copy(index_free)
        self.index_fixp_target = np.copy(index_fixp_target)
        self.index_fixp_source = np.copy(index_fixp_source)

    @classmethod
    def from_leastsquare(
        cls, output: OptimizeResult, chi2: float, dof: float
    ) -> Solution:
        '''Construct Solution() from output of least_square.
        '''
        p_bestfit = output.x
        J = output.jac
        # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
        cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
        result_error = np.sqrt(np.diag(cov))
        p_bestfit = restore_params(p_bestfit)
        result_error = restore_params(result_error)
        return Solution(
            p_bestfit,
            result_error,
            result_error,
            chi2,
            dof,
            cov,
            mode_fitting='leastsquare',
        )

    @classmethod
    def from_sampler(
        cls,
        sampler: emcee.EnsembleSampler,
        dof: float,
        func_fit: Callable,
        # mask_for_fit: Optional[np.ndarray] = None,
    ) -> Solution:
        '''Construct Solution() from sampler.
        '''
        try:
            tau = sampler.get_autocorr_time()
            burnin = int(np.max(tau) * 2.0)
            thin = int(np.min(tau) / 2.0)
        except emcee.autocorr.AutocorrError:
            c.logger.warning(
                'MCMC may not be converged.'
                'Please be careful to use the best-fit parameters.'
            )
            # HACK: these estimates may be wrong.
            shape = sampler.get_chain(discard=0, thin=1).shape
            burnin = int(shape[0] / 100.0 * 2.0)
            thin = int(shape[0] / 100.0 / 2.0)
        flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

        p16, p50, p84 = np.percentile(flat_samples, [16, 50, 84], axis=0)
        best = restore_params(p50)
        error_high = restore_params(p84 - p50)
        error_low = restore_params(p50 - p16)

        chi2 = np.sum(calculate_chi(best, func_fit) ** 2.0)
        return cls(
            best,
            error_high,
            error_low,
            chi2,
            dof,
            np.array(0.0),
            mode_fitting='mcmc',
            sampler=sampler,
        )

    @classmethod
    def from_montecarlo(cls, params: np.ndarray, chi2: float, dof: float) -> Solution:
        '''Construct Solution() from montecarlo perturbations.
        '''
        p16, p50, p84 = np.percentile(params, [16, 50, 84], axis=0)
        best = restore_params(p50)
        error_high = restore_params(p84 - p50)
        error_low = restore_params(p50 - p16)
        return cls(
            best,
            error_high,
            error_low,
            0.0,
            0.0,
            np.array(0.0),
            params_mc=params,
            mode_fitting='montecarlo',
        )

    def set_metainfo(
        self, z: Optional[float] = None, header: Optional[fits.Header] = None
    ) -> None:
        '''Set meta infomation used in Solution
        '''
        keys_arguments = ['z', 'header']

        for k in keys_arguments:
            if (value_input := locals()[k]) is not None:
                setattr(self.meta, k, value_input)

    @dataclass
    class MetaInfoOfSolution:
        '''Meta data container of Solution.
        '''

        z: float = 0.0
        header: Optional[fits.Header] = None

    def add_units(
        self, params: Optional[Union[InputParams, np.ndarray]] = None
    ) -> FitParamsWithUnits:
        '''Get best parameters with physical units.
        '''
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
        '''Restore parameters with pfix by inserting parameters into params.
        '''
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


class InputParams(NamedTuple):
    '''Input parameters for construct_model_at_imageplane.
    '''

    x0_dyn: float  #: the coordinate on x-axis
    y0_dyn: float
    PA_dyn: float
    inclination_dyn: float
    radius_dyn: float
    velocity_sys: float
    mass_dyn: float
    brightness_center: float
    velocity_dispersion: float
    radius_emi: float
    x0_emi: float
    y0_emi: float
    PA_emi: float
    inclination_emi: float

    def to_units(
        self, header: fits.Header, redshift: float = 0.0
    ) -> FitParamsWithUnits:
        '''Return input parameters with units.
        '''
        return FitParamsWithUnits.from_inputparams(self, header, redshift)


class InputParamsArray(NamedTuple):
    '''Input parameter array for construct_model_at_imageplane.
    '''

    x0_dyn: np.ndarray
    y0_dyn: np.ndarray
    PA_dyn: np.ndarray
    inclination_dyn: np.ndarray
    radius_dyn: np.ndarray
    velocity_sys: np.ndarray
    mass_dyn: np.ndarray
    brightness_center: np.ndarray
    velocity_dispersion: np.ndarray
    radius_emi: np.ndarray
    x0_emi: np.ndarray
    y0_emi: np.ndarray
    PA_emi: np.ndarray
    inclination_emi: np.ndarray

    def to_units(
        self, header: fits.Header, redshift: float = 0.0
    ) -> FitParamsWithUnits:
        '''Return input parameters with units.
        '''
        return FitParamsWithUnits.from_inputparams(self, header, redshift)

    @classmethod
    def from_ndarray(cls, params: np.ndarray):
        pass


@dataclass
class FitParamsWithUnits:
    '''Fitting parameters with units.
    '''

    x0_dyn: u.Quantity
    y0_dyn: u.Quantities
    PA_dyn: u.Quantities
    inclination_dyn: u.Quantities
    radius_dyn: u.Quantities
    velocity_sys: u.Quantities
    mass_dyn: u.Quantities
    brightness_center: u.Quantities
    velocity_dispersion: u.Quantities
    radius_emi: u.Quantities
    x0_emi: u.Quantities
    y0_emi: u.Quantities
    PA_emi: u.Quantities
    inclination_emi: u.Quantities
    header: Optional[fits.Header] = field(default=None, repr=False)
    z: float = field(default=0.0, repr=False)
    wcs: Optional[WCS] = field(init=False, repr=False)
    pixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
    freq_rest: Optional[u.Quantities] = field(init=False, repr=False)
    vpixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
    diskmassscale: Optional[u.Equivalency] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.header:
            self.wcs = WCS(self.header)
            self.freq_rest = self.header['RESTFRQ'] * u.Hz

            deg_pix = abs(self.header['CDELT1']) * u.Unit(self.header['CUNIT1']) / u.pix
            self.pixelscale = misc.pixel_scale(deg_pix.to(u.arcsec / u.pix), self.z)

            dfreq_pix = abs(self.header['CDELT3']) * u.Unit(self.header['CUNIT3'])
            opt_equiv = u.doppler_optical(self.freq_rest)
            dv_pix = (self.freq_rest - dfreq_pix).to(u.km / u.s, opt_equiv)
            self.vpixelscale = misc.vpixel_scale(dv_pix / u.pix)

            self.diskmassscale = (
                misc.diskmass_scale(self.pixelscale, self.vpixelscale)
                if self.z > 0.0
                else None
            )

        else:
            self.wcs = None
            self.pixelscale = None
            self.freq_rest = None
            self.vpixelscale = None
            self.diskmassscale = None

    def to_physicalscale(self) -> None:
        '''Convert values to physicalscales.
        '''
        if self.header is None:
            raise ValueError('header is not input.')
        assert isinstance(self.wcs, WCS)

        wcs_celestial = self.wcs.celestial
        wcs_spectral = self.wcs.spectral
        coord_celestial = wcs_celestial.pixel_to_world(
            [self.x0_dyn, self.x0_emi], [self.y0_dyn, self.y0_emi]
        )
        coord_spectral = wcs_spectral.pixel_to_world(self.velocity_sys)
        coord_spectral_kms = coord_spectral.to(
            u.km / u.s, doppler_convention='optical', doppler_rest=self.freq_rest
        )
        self.x0_dyn = coord_celestial.ra[0]
        self.y0_dyn = coord_celestial.dec[0]
        self.x0_emi = coord_celestial.ra[1]
        self.y0_emi = coord_celestial.dec[1]
        self.velocity_sys = coord_spectral_kms.quantity

        self.radius_dyn = self.radius_dyn.to(u.arcsec, self.pixelscale)
        self.radius_emi = self.radius_emi.to(u.arcsec, self.pixelscale)
        self.brightness_center = self.brightness_center.to(
            u.Jy / u.arcsec ** 2, self.pixelscale
        )
        self.velocity_dispersion = self.velocity_dispersion.to(
            u.km / u.s, self.vpixelscale
        )

        if self.z > 0.0:
            self.radius_dyn = self.radius_dyn.to(u.kpc, self.pixelscale)
            self.radius_emi = self.radius_emi.to(u.kpc, self.pixelscale)
            self.mass_dyn = self.mass_dyn.physical.to(u.Msun, self.diskmassscale)

    def vmax(self):
        '''Maximum rotation velcoity.
        '''
        return func.maximum_rotation_velocity(self.mass_dyn, self.radius_dyn)

    @classmethod
    def from_inputparams(
        cls,
        inputparams: Union[InputParams, InputParamsArray],
        header: Optional[fits.Header] = None,
        z: float = 0.0,
    ) -> FitParamsWithUnits:
        '''Constructer from InputParams
        '''
        dictionary = inputparams._asdict()
        input_dict = {}
        units = (
            u.dimensionless_unscaled,
            u.dimensionless_unscaled,
            u.rad,
            u.rad,
            u.pix,
            u.pix,
            u.dex(u.pix ** 3),
            u.Jy / u.pix / u.pix,
            u.pix,
            u.pix,
            u.dimensionless_unscaled,
            u.dimensionless_unscaled,
            u.rad,
            u.rad,
        )
        for (key, value), unit in zip(dictionary.items(), units):
            input_dict[key] = value * unit

        if header is None:
            return cls(**input_dict)
        else:
            clsself = cls(header=header, z=z, **input_dict)
            clsself.to_physicalscale()
            return clsself


def get_bound_params(
    x0_dyn: tuple[float, float] = (-np.inf, np.inf),
    y0_dyn: tuple[float, float] = (-np.inf, np.inf),
    PA_dyn: tuple[float, float] = (0.0, 2 * np.pi),
    inclination_dyn: tuple[float, float] = (0.0, np.pi / 2),
    radius_dyn: tuple[float, float] = (0.0, np.inf),
    velocity_sys: tuple[float, float] = (-np.inf, np.inf),
    mass_dyn: tuple[float, float] = (-np.inf, np.inf),
    brightness_center: tuple[float, float] = (0.0, np.inf),
    velocity_dispersion: tuple[float, float] = (0.0, np.inf),
    radius_emi: tuple[float, float] = (0.0, np.inf),
    x0_emi: tuple[float, float] = (-np.inf, np.inf),
    y0_emi: tuple[float, float] = (-np.inf, np.inf),
    PA_emi: tuple[float, float] = (0.0, 2 * np.pi),
    inclination_emi: tuple[float, float] = (0.0, np.pi / 2),
) -> tuple[InputParams, InputParams]:
    '''Return bound parameters.
    '''

    def _bound(i: int) -> InputParams:
        return InputParams(
            x0_dyn=x0_dyn[i],
            y0_dyn=y0_dyn[i],
            PA_dyn=PA_dyn[i],
            inclination_dyn=inclination_dyn[i],
            radius_dyn=radius_dyn[i],
            velocity_sys=velocity_sys[i],
            mass_dyn=mass_dyn[i],
            brightness_center=brightness_center[i],
            velocity_dispersion=velocity_dispersion[i],
            radius_emi=radius_emi[i],
            x0_emi=x0_emi[i],
            y0_emi=y0_emi[i],
            PA_emi=PA_emi[i],
            inclination_emi=inclination_emi[i],
        )

    lower, upper = (0, 1)
    return (_bound(lower), _bound(upper))


def is_init_outside_of_bound(
    init: tuple[float, ...], bound: tuple[tuple[float, ...], tuple[float, ...]]
) -> bool:
    '''Return True if init is outside of bound.
    '''
    bound0, bound1 = bound
    for i, b0, b1 in zip(init, bound0, bound1):
        if (i < b0) | (b1 < i):
            return True
    return False


def initialize_globalparameters_for_image(
    datacube: DataCube,
    mask_for_fit: np.ndarray,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    beam_vis: Optional[np.ndarray] = None,
) -> None:
    '''Set global parameters used in fitting.py in the image plane.
    '''
    global cube, cube_error, xx_grid, yy_grid, vv_grid
    global lensing, lensing_interpolation, convolve, mask

    cube = np.copy(datacube.imageplane)
    cube_error = datacube.rms()
    cube_error = cube_error[:, np.newaxis, np.newaxis]
    mask = mask_for_fit
    if not np.any(mask_for_fit):
        raise ValueError(
            '"mask_for_fit" filled by False. We believe that you don\'t want to use it.'
        )
    cube_error = np.broadcast_to(cube_error, cube.shape)
    cube = cube[mask]
    cube_error = cube_error[mask]

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = misc.no_convolve
    f_no_lensing: Callable = misc.no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing

    vv_grid, yy_grid_image, xx_grid_image = datacube.coord_imageplane
    yy_grid = yy_grid_image[[0], :, :]
    xx_grid = xx_grid_image[[0], :, :]
    xx_grid, yy_grid = lensing(xx_grid, yy_grid)
    vv_grid = vv_grid[:, 0, 0].reshape(-1, 1, 1)

    lensing_interpolation = (
        func_create_lensinginterp(xx_grid_image, yy_grid_image)
        if func_create_lensinginterp
        else misc.no_lensing_interpolation
    )


def initialize_globalparameters_for_uv(
    datacube: DataCube,
    beam_vis: np.ndarray,
    norm_weight: float,
    mask_for_fit: np.ndarray,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
) -> None:
    '''Set global parameters used in fitting.py in the uv plane.
    '''
    global cube, cube_error, cubeshape, xx_grid, yy_grid, vv_grid, xslice, yslice
    global lensing, lensing_interpolation, mask

    size = datacube.original[0, :, :].size  # constant var needed for convolution
    cube = datacube.uvplane / beam_vis / size
    cube_error = np.sqrt(abs(beam_vis.real)) / beam_vis / np.sqrt(norm_weight) / size
    cube_error = _correct_cube_error_for_uv(cube_error)
    cubeshape = datacube.original[datacube.vslice, :, :].shape
    # sigma = (cube / cube_error).real
    # mask_to_remove_outlier = (sigma > -5) & (sigma < 5)
    # if mask_for_fit is not None:
    #     mask = mask_for_fit & mask_to_remove_outlier
    # else:
    #     mask = mask_to_remove_outlier

    mask = mask_for_fit
    if not np.any(mask):
        raise ValueError(
            'We believe that you don\'t want to use "mask_for_fit" filled by False.'
        )
    mask = _add_mask_for_uv(mask)
    cube_error = np.broadcast_to(cube_error, cube.shape)
    cube = cube[mask]
    cube_error = cube_error[mask]
    # shape_data = datacube.uvplane.shape
    # xarray = np.arange(0, shape_data[2])
    # yarray = np.arange(0, shape_data[1])
    # varray = np.arange(datacube.vlim[0], datacube.vlim[1])
    # vv_grid, yy_grid, xx_grid = np.meshgrid(varray, yarray, xarray, indexing='ij')

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_lensing: Callable = misc.no_lensing
    lensing = func_lensing if func_lensing else f_no_lensing

    vv_grid, yy_grid_image, xx_grid_image = datacube.coord_imageplane
    yy_grid = yy_grid_image[[0], :, :]
    xx_grid = xx_grid_image[[0], :, :]
    xx_grid, yy_grid = lensing(xx_grid, yy_grid)
    vv_grid = vv_grid[:, 0, 0].reshape(-1, 1, 1)
    xslice, yslice = (datacube.xslice, datacube.yslice)

    # beam_visibility = beam_vis
    # mask_FoV = datacube.mask_FoV[datacube.vslice, :, :]

    if func_create_lensinginterp:
        lensing_interpolation = func_create_lensinginterp(xx_grid_image, yy_grid_image)
    else:
        lensing_interpolation = misc.no_lensing_interpolation


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


class FixParams(NamedTuple):
    '''Fixed parameters for construct_model_at_imageplane.
    '''

    x0_dyn: Optional[float] = None
    y0_dyn: Optional[float] = None
    PA_dyn: Optional[float] = None
    inclination_dyn: Optional[float] = None
    radius_dyn: Optional[float] = None
    velocity_sys: Optional[float] = None
    mass_dyn: Optional[float] = None
    brightness_center: Optional[float] = None
    velocity_dispersion: Optional[float] = None
    radius_emi: Optional[Union[float, bool]] = None
    x0_emi: Optional[Union[float, bool]] = None
    y0_emi: Optional[Union[float, bool]] = None
    PA_emi: Optional[Union[float, bool]] = None
    inclination_emi: Optional[Union[float, bool]] = None


def set_fixedparameters(fix: Optional[FixParams], is_separate: bool) -> None:
    '''Set global parameters related with fixed parameters.

    if a value in fix is:
    - None: the parameter is not fixed
    - float: the perameter is fixed to the float value
    - True: the parameter has the same value as another parameter
    '''
    global parameters_preset, index_free, index_fixp_target, index_fixp_source
    parameters_preset = np.empty(14)
    index_free = []
    index_fixp_target = []
    index_fixp_source = []

    parameters_fixp = FixParams(
        radius_emi=4, x0_emi=0, y0_emi=1, PA_emi=2, inclination_emi=3
    )
    free_parameter = None
    fixed_to_another_parameter = True

    if (fix is None) and (is_separate):
        parameters_preset = None
        return
    elif fix is None:
        _fix = FixParams()
    else:
        _fix = fix

    if is_separate is False:
        _fix = _fix._replace(
            radius_emi=True, x0_emi=True, y0_emi=True, PA_emi=True, inclination_emi=True
        )

    for i, p in enumerate(_fix):
        p_is_fixed_to_a_value = isinstance(p, float)

        if p_is_fixed_to_a_value:
            parameters_preset[i] = p

        if p is free_parameter:
            index_free.append(i)

        if p is fixed_to_another_parameter:
            if (idx := parameters_fixp[i]) is None:
                raise TypeError(
                    f'An unsupported parameter p[{i}] is set to True in FixParams.'
                )
            index_fixp_target.append(i)
            index_fixp_source.append(int(idx))


def restore_params(params: tuple[float, ...]) -> tuple[float, ...]:
    '''Restore parameters with pfix by inserting parameters into params.
    - params -- parameter array. Its length is shorter than 14, which is the
                total number of parameters.
    '''
    global parameters_preset, index_free, index_fixp_target, index_fixp_source
    if (parameters_preset is None) or (len(params) == 14):
        return params
    parameters_preset[index_free] = params
    parameters_preset[index_fixp_target] = parameters_preset[index_fixp_source]
    return tuple(parameters_preset)


def shorten_init_and_bound_ifneeded(
    init: Sequence[float], bound: tuple[Sequence[float], Sequence[float]]
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], tuple[float, ...]]]:
    '''Shorten init and bound parameter to match appropreate lengths.
    '''
    global index_free
    new_init = tuple(np.array(init)[index_free])
    new_bound0 = tuple(np.array(bound[0])[index_free])
    new_bound1 = tuple(np.array(bound[1])[index_free])
    return new_init, (new_bound0, new_bound1)


def initialguess(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    is_separate: bool = False,
) -> InputParams:
    '''Guess initial parameters by fitting moment 0 and 1 maps.
    '''
    param0 = least_square_moment0(
        datacube, func_convolve, func_lensing, func_create_lensinginterp
    )

    p = param0
    vcen = np.sum(datacube.vlim) / 2.0
    init = [p[3], p[4], 1.0, vcen, p[2], p[0], p[1]]

    param1 = least_square_moment1(
        datacube, init, func_convolve, func_lensing, func_create_lensinginterp
    )

    if not is_separate:
        param1[5] = param0[0]
        param1[6] = param0[1]
        param0[2] = param1[4]  # PA should come from dynamics...?
        param1[0] = param0[3]
        param1[1] = param0[4]

    return InputParams(
        x0_dyn=param1[5],
        y0_dyn=param1[6],
        PA_dyn=param1[4],
        inclination_dyn=param1[0],
        radius_dyn=param1[1],
        velocity_sys=param1[3],
        mass_dyn=param1[2],
        brightness_center=param0[5],
        velocity_dispersion=1.5,
        radius_emi=param0[4],
        x0_emi=param0[0],
        y0_emi=param0[1],
        PA_emi=param0[2],
        inclination_emi=param0[3],
    )


def least_square_moment0(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    mask_use: Optional[np.ndarray] = None,
) -> list[float]:
    '''Least square fitting of moment 0 map.

    This function is mainly for guessing initial parameters formain fitting routine.
    '''
    if mask_use is None:
        mask_use = np.ones_like(datacube.moment0()).astype(bool)[None, :, :]
    initialize_globalparameters_for_moment(
        datacube,
        mask_use,
        func_convolve,
        func_lensing,
        func_create_lensinginterp,
        mom=0,
    )
    func_fit = construct_model_moment0

    x0, y0 = datacube.xgrid.mean(), datacube.ygrid.mean()
    brightness0 = datacube.moment0().max()
    init = (x0, y0, np.pi / 2, np.pi / 4, 1.0, brightness0)
    bound = (
        (-np.inf, -np.inf, 0, 0, 0, 0),
        (np.inf, np.inf, np.pi, 0.5 * np.pi, np.inf, np.inf),
    )
    args = (func_fit,)
    output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
    return output.x


def least_square_moment1(
    datacube: DataCube,
    init: Sequence[float],
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
) -> list[float]:
    '''Least square fitting of moment 1 map.

    This function is mainly for guessing initial parameters formain fitting routine.
    '''
    rms = datacube.rms_moment0()
    moment1 = datacube.pixmoment1(thresh=3 * rms)
    mask = np.isfinite(moment1)

    initialize_globalparameters_for_moment(
        datacube, mask, func_convolve, func_lensing, func_create_lensinginterp, mom=1,
    )
    func_fit = construct_model_moment1

    bound = (
        (0, 0, -np.inf, -np.inf, 0, -np.inf, -np.inf),
        (0.5 * np.pi, np.inf, np.inf, np.inf, 2 * np.pi, np.inf, np.inf),
    )

    args = (func_fit,)
    output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
    return output.x


def initialize_globalparameters_for_moment(
    datacube: DataCube,
    mask_for_fit: np.ndarray,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    mom: int = 0,
) -> None:
    '''Set global parameters used in fitting.py.
    '''
    global cube, cube_error, xx_grid, yy_grid
    global lensing, lensing_interpolation, convolve, mask

    mask = mask_for_fit
    if mom == 0:
        cube = datacube.moment0()[mask.squeeze()]
        cube_error = np.array(datacube.rms_moment0())
    elif mom == 1:
        rms = datacube.rms_moment0()
        cube = datacube.pixmoment1(thresh=3 * rms)
        idx = np.isfinite(cube) & mask.squeeze()
        cube = cube[idx]  # cube becomes 1d
        mom0 = datacube.moment0()[idx]
        cube_error = 1 / np.sqrt(mom0)
    _, yy_grid_image, xx_grid_image = datacube.coord_imageplane
    xx_grid = xx_grid_image[0, :, :]
    yy_grid = yy_grid_image[0, :, :]
    xx_grid, yy_grid = lensing(xx_grid, yy_grid)

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = misc.no_convolve
    f_no_lensing: Callable = misc.no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing
    lensing_interpolation = (
        func_create_lensinginterp(xx_grid_image, yy_grid_image)
        if func_create_lensinginterp
        else misc.no_lensing_interpolation
    )


def map_globals_to_childprocesses(pool: Pool) -> None:
    '''Map parent global parameters to parameters in pooled child processes.
    '''
    keys_globals = [
        'cube',
        'cube_error',
        'cubeshape',
        'xx_grid',
        'yy_grid',
        'vv_grid',
        'xslice',
        'yslice',
        'lensing',
        'lensing_interpolation',
        'convolve',
        'mask',
        # 'mask_FoV',
        'parameters_preset',
        'index_free',
        'index_fixp_target',
        'index_fixp_source',
    ]
    parent_globals = {}

    for k in keys_globals:
        try:
            parent_globals[k] = globals()[k]
        except KeyError:
            pass

    # fn_pkl = f'tokult_map_globals_to_childprocesses-{time.time()}.pkl'
    # with open(fn_pkl, 'wb') as f:
    #     pickle.dump(parent_globals, f)
    # NOTE: need to use private attributes to know # of processes used in pool.
    pool.map(_set_globals_in_process, [parent_globals] * pool._processes)  # type:ignore
    # Path(fn_pkl).unlink()


def _set_globals_in_process(parent_params: dict) -> None:
    '''Set global parameters in a child process.'''
    # with open(fn_pkl, 'rb') as f:
    #     parent_params = pickle.load(f)
    for key, value in parent_params.items():
        globals()[key] = value
