'''Modules of fitting functions
'''
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares as sp_least_squares
from typing import Callable, Sequence, Optional, Union, TYPE_CHECKING
from typing import NamedTuple
from numpy.typing import ArrayLike
from . import function as func
from .misc import rotate_coord, polar_coord, no_lensing, no_convolve, fft2

if TYPE_CHECKING:
    from .core import DataCube

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
beam_visibility: np.ndarray
xx_grid: np.ndarray
yy_grid: np.ndarray
vv_grid: np.ndarray
xslice: slice
yslice: slice
lensing: Callable = no_lensing
convolve: Callable = no_convolve

# To fix parameters in fitting
parameters_preset: Optional[np.ndarray] = None
index_free: list[int]
index_fixp_target: list[int]
index_fixp_source: list[int]


def least_square(
    datacube: DataCube,
    init: Sequence[float],
    bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    beam_vis: Optional[np.ndarray] = None,
    mask_use: Optional[ArrayLike] = None,
    niter: int = 1,
    mode_fit: str = 'image',
    fix: Optional[FixParams] = None,
    is_separate: bool = False,
) -> Solution:
    '''Least square fitting using scipy.optimize.least_squares
    '''
    if mode_fit == 'image':
        initialize_globalparameters_for_image(datacube, func_convolve, func_lensing)
        func_fit = construct_convolvedmodel
    elif mode_fit == 'uv':
        if beam_vis is None:
            raise ValueError('"beam_vis" is necessarily for uvfit')
        initialize_globalparameters_for_uv(datacube, beam_vis, func_lensing)
        func_fit = construct_uvmodel
    else:
        raise ValueError(f'mode_fit is "image" or "uv", no option for "{mode_fit}".')

    set_fixedparameters(fix, is_separate)
    bound = get_bound_params() if bound is None else bound
    _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
    args = (func_fit, mask_use)

    print('run fitting')
    for _ in range(niter):
        output = sp_least_squares(calculate_chi, _init, args=args, bounds=_bound)
        _init = output.x
    print('end fitting')

    p_bestfit = output.x
    dof = datacube.imageplane.size - 1 - len(init)
    chi2 = np.sum(calculate_chi(p_bestfit, func_fit) ** 2.0)
    J = output.jac
    # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
    cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
    result_error = np.sqrt(np.diag(cov))
    p_bestfit = restore_params(p_bestfit)
    result_error = restore_params(result_error)
    return Solution(p_bestfit, result_error, chi2, dof, cov)


def calculate_chi(
    params: tuple[float, ...], model_func: Callable, index: Optional[ArrayLike] = None,
) -> np.ndarray:
    '''Calcurate chi = (data-model)/error for least square fitting
    '''
    model = model_func(params)
    if index is not None:
        model = model[index]

        # # Choose fitting region via indices.
        # iv = (v - dv * c.conf.chan_start - vel_start) / dv
        # iv = np.round(iv).astype(int)
        # ix = (x - xlow).astype(int)
        # iy = (y - ylow).astype(int)
        # model_last = model[iy, ix, iv]

    chi = abs(cube - model) / cube_error
    return chi.ravel()


def construct_convolvedmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    model = construct_model_at_imageplane(params)
    model_convolved = np.empty_like(model)
    for i, image in enumerate(model):
        model_convolved[i, :, :] = convolve(image, index=i)
    return model_convolved


def construct_uvmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    global cube, yslice, xslice, beam_visibility
    model_cutout = construct_model_at_imageplane(params)
    model_image = np.zeros_like(cube)
    model_image[:, yslice, xslice] = model_cutout
    model_visibility = fft2(model_image)
    return model_visibility * beam_visibility


def construct_model_at_imageplane(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube on image plane using parameters and the grav. lensing.
    '''
    _params = restore_params(params)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = _params

    # velocity field
    coord_image_v = np.moveaxis(np.array([xx_grid - p0, yy_grid - p1]), 0, -1)
    rr, pphi = to_objectcoord_from(coord_image_v, PA=p2, incl=p3)
    velocity = p5 + func.freeman_disk(rr, pphi, mass_dyn=p6, rnorm=p4, incl=p3)

    # spatial intensity distribution
    coord_image_i = np.moveaxis(np.array([xx_grid - p10, yy_grid - p11]), 0, -1)
    rr_i, _ = to_objectcoord_from(coord_image_i, PA=p12, incl=p13)
    intensity = func.reciprocal_exp(rr_i, norm=p7, rnorm=p9)

    # create cube
    model = func.gaussian(vv_grid, center=velocity, sigma=p8, area=intensity)
    return model


def to_objectcoord_from(
    coord_image: np.ndarray, PA: float, incl: float
) -> tuple[np.ndarray, np.ndarray]:
    '''Convert coordinates from imageplane to object polar coordinates.
    '''
    pa = PA
    inclination = incl

    coord_source = lensing(coord_image)
    coord_object = rotate_coord(coord_source, pa)
    xx, yy = np.moveaxis(coord_object, -1, 0)
    yy = yy / np.cos(inclination)
    r, phi = polar_coord(xx, yy)
    return r, phi


def construct_model_at_imageplane_with(
    params: tuple[float, ...],
    xx_grid: np.ndarray,
    yy_grid: np.ndarray,
    vv_grid: np.ndarray,
    lensing: Optional[Callable] = None,
) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    lensing = no_lensing if lensing is None else lensing
    keys_globals = [
        'xx_grid',
        'yy_grid',
        'vv_grid',
        'lensing',
    ]
    _globals = {}

    for k in keys_globals:
        _globals[k] = globals()[k]
        globals()[k] = locals()[k]

    model = construct_model_at_imageplane(params)

    for k in keys_globals:
        globals()[k] = _globals[k]

    return model


def construct_model_moment0(params: list[float]) -> np.ndarray:
    '''Construct a model moment0 map convolved with dirtybeam.
    '''
    p0, p1, p2, p3, p4, p5 = params

    coord_image_i = np.moveaxis(np.array([xx_grid - p0, yy_grid - p1]), 0, -1)
    rr_i, _ = to_objectcoord_from(coord_image_i, PA=p2, incl=p3)
    intensity = func.reciprocal_exp(rr_i, norm=p5, rnorm=p4)
    model = convolve(intensity, index=0)

    return model


def construct_model_moment1(params: list[float]) -> np.ndarray:
    '''Construct a model moment1 map convolved with dirtybeam.
    '''
    p0, p1, p2, p3, p4, p5, p6 = params

    coord_image_v = np.moveaxis(np.array([xx_grid - p5, yy_grid - p6]), 0, -1)
    rr, pphi = to_objectcoord_from(coord_image_v, PA=p4, incl=p0)
    velocity = p3 + func.freeman_disk(rr, pphi, mass_dyn=p2, rnorm=p1, incl=p0)
    # # NOTE: convolving velocity is correct?
    # model = convolve(velocity, index=0)

    return velocity


class Solution:
    '''Contains output solutions of fittings.
    '''

    def __init__(
        self,
        p_best: list[float],
        error: list[float],
        chi2: float,
        dof: float,
        cov: np.ndarray,
    ) -> None:
        self.best = InputParams(*p_best)
        self.error = InputParams(*error)
        self.chi2 = chi2
        self.dof = dof
        self.cov = cov


class InputParams(NamedTuple):
    '''Input parameters for construct_model_at_imageplane.
    '''

    x0_dyn: float
    y0_dyn: float
    PA_dyn: float
    incliation_dyn: float
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


def get_bound_params(
    x0_dyn: tuple[float, float] = (-np.inf, np.inf),
    y0_dyn: tuple[float, float] = (-np.inf, np.inf),
    PA_dyn: tuple[float, float] = (0.0, 2 * np.pi),
    incliation_dyn: tuple[float, float] = (0.0, np.pi / 2),
    radius_dyn: tuple[float, float] = (0.0, np.inf),
    velocity_sys: tuple[float, float] = (-np.inf, np.inf),
    mass_dyn: tuple[float, float] = (0.0, np.inf),
    brightness_center: tuple[float, float] = (0.0, np.inf),
    velocity_dispersion: tuple[float, float] = (0.0, np.inf),
    radius_emi: tuple[float, float] = (0.0, np.inf),
    x0_emi: tuple[float, float] = (-np.inf, np.inf),
    y0_emi: tuple[float, float] = (-np.inf, np.inf),
    PA_emi: tuple[float, float] = (0.0, np.pi),
    inclination_emi: tuple[float, float] = (0.0, np.pi),
) -> tuple[InputParams, InputParams]:
    '''Return bound parameters.
    '''

    def _bound(i: int) -> InputParams:
        return InputParams(
            x0_dyn=x0_dyn[i],
            y0_dyn=y0_dyn[i],
            PA_dyn=PA_dyn[i],
            incliation_dyn=incliation_dyn[i],
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


def initialize_globalparameters_for_image(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    beam_vis: Optional[np.ndarray] = None,
) -> None:
    '''Set global parameters used in fitting.py in the image plane.
    '''
    global cube, cube_error, xx_grid, yy_grid, vv_grid
    global lensing, convolve

    cube = np.copy(datacube.imageplane)
    cube_error = datacube.rms()
    cube_error = cube_error[:, np.newaxis, np.newaxis]
    vv_grid, yy_grid, xx_grid = datacube.coord_imageplane

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = no_convolve
    f_no_lensing: Callable = no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing


def initialize_globalparameters_for_uv(
    datacube: DataCube, beam_vis: np.ndarray, func_lensing: Optional[Callable] = None,
) -> None:
    '''Set global parameters used in fitting.py in the uv plane.
    '''
    global cube, cube_error, xx_grid, yy_grid, vv_grid, xslice, yslice
    global lensing, beam_visibility

    cube = np.copy(datacube.uvplane)
    cube_error = np.array(1.0)
    # shape_data = datacube.uvplane.shape
    # xarray = np.arange(0, shape_data[2])
    # yarray = np.arange(0, shape_data[1])
    # varray = np.arange(datacube.vlim[0], datacube.vlim[1])
    # vv_grid, yy_grid, xx_grid = np.meshgrid(varray, yarray, xarray, indexing='ij')
    vv_grid, yy_grid, xx_grid = datacube.coord_imageplane
    xslice, yslice = (datacube.xslice, datacube.yslice)
    beam_visibility = beam_vis

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_lensing: Callable = no_lensing
    lensing = func_lensing if func_lensing else f_no_lensing


class FixParams(NamedTuple):
    '''Fixed parameters for construct_model_at_imageplane.
    '''

    x0_dyn: Optional[float] = None
    y0_dyn: Optional[float] = None
    PA_dyn: Optional[float] = None
    incliation_dyn: Optional[float] = None
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
) -> InputParams:
    '''Guess initial parameters by fitting moment 0 and 1 maps.
    '''
    param0 = least_square_moment0(datacube, func_convolve, func_lensing)

    p = param0
    vcen = np.sum(datacube.vlim) / 2.0
    init = [p[3], p[4], 10.0, vcen, p[2], p[0], p[1]]

    param1 = least_square_moment1(datacube, init, func_convolve, func_lensing)

    return InputParams(
        x0_dyn=param1[5],
        y0_dyn=param1[6],
        PA_dyn=param1[4],
        incliation_dyn=param1[0],
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
    mask_use: Optional[ArrayLike] = None,
) -> list[float]:
    '''Least square fitting of moment 0 map.

    This function is mainly for guessing initial parameters formain fitting routine.
    '''
    initialize_globalparameters_for_moment(datacube, func_convolve, func_lensing, mom=0)
    func_fit = construct_model_moment0

    x0, y0 = datacube.xgrid.mean(), datacube.ygrid.mean()
    init = (x0, y0, np.pi / 2, np.pi / 4, 1.0, 3.0)
    bound = (
        (-np.inf, -np.inf, 0, 0, 0, 0),
        (np.inf, np.inf, np.pi, 0.5 * np.pi, np.inf, np.inf),
    )
    args = (func_fit, mask_use)
    output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
    return output.x


def least_square_moment1(
    datacube: DataCube,
    init: Sequence[float],
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
) -> list[float]:
    '''Least square fitting of moment 1 map.

    This function is mainly for guessing initial parameters formain fitting routine.
    '''
    initialize_globalparameters_for_moment(datacube, func_convolve, func_lensing, mom=1)
    func_fit = construct_model_moment1

    bound = (
        (0, 0, 0, -np.inf, 0, -np.inf, -np.inf),
        (0.5 * np.pi, np.inf, np.inf, np.inf, 2 * np.pi, np.inf, np.inf),
    )

    rms = datacube.rms_moment0()
    moment1 = datacube.pixmoment1(thresh=3 * rms)
    mask = np.isfinite(moment1)

    args = (func_fit, mask)
    output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
    return output.x


def initialize_globalparameters_for_moment(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    mom: int = 0,
) -> None:
    '''Set global parameters used in fitting.py.
    '''
    global cube, cube_error, xx_grid, yy_grid, lensing, convolve

    if mom == 0:
        cube = datacube.moment0()
        cube_error = datacube.rms_moment0()
    elif mom == 1:
        rms = datacube.rms_moment0()
        cube = datacube.pixmoment1(thresh=3 * rms)
        idx = np.isfinite(cube)
        cube = cube[idx]  # cube becomes 1d
        mom0 = datacube.moment0()[idx]
        cube_error = 1 / np.sqrt(mom0)
    _, yy_grid, xx_grid = datacube.coord_imageplane
    xx_grid = xx_grid[0, :, :]
    yy_grid = yy_grid[0, :, :]

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = no_convolve
    f_no_lensing: Callable = no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing
