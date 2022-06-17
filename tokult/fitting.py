'''Modules of fitting functions
'''
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares as sp_least_squares
from typing import Callable, Sequence, Optional, TYPE_CHECKING
from typing import NamedTuple
from numpy.typing import ArrayLike
from . import function as func
from .misc import rotate_coord, polar_coord, no_lensing, no_convolve

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

imagecube: np.ndarray
imagecube_error: np.ndarray
xx_grid: np.ndarray
yy_grid: np.ndarray
vv_grid: np.ndarray
lensing: Callable = no_lensing
convolve: Callable = no_convolve


def least_square(
    datacube: DataCube,
    init: Sequence[float],
    bound: Optional[ArrayLike] = None,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    mask_use: Optional[ArrayLike] = None,
    niter: int = 1,
) -> Solution:
    '''Least square fitting using scipy.optimize.least_squares
    '''
    initialize_globalparameters(datacube, func_convolve, func_lensing)
    func_fit = construct_model_convolved
    bound = get_bound_params() if bound is None else bound

    args = (func_fit, mask_use)
    for _ in range(niter):
        output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
        init = output.x

    p_bestfit = output.x
    dof = datacube.imageplane.size - 1 - len(init)
    chi2 = np.sum(calculate_chi(p_bestfit, func_fit) ** 2.0)
    J = output.jac
    # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
    cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
    result_error = np.sqrt(np.diag(cov))
    return Solution(p_bestfit, result_error, chi2, dof, cov)


def calculate_chi(
    params: list[float], model_func: Callable, index: Optional[ArrayLike] = None,
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

    chi = (imagecube - model) / imagecube_error
    return chi.ravel()


def construct_model_convolved(params: list[float]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.
    '''
    model = construct_model_at_imageplane(params)
    model_convolved = np.empty_like(model)
    for i, image in enumerate(model):
        model_convolved[i, :, :] = convolve(image, index=i)

    return model_convolved


def construct_model_at_imageplane(params: list[float]) -> np.ndarray:
    '''Construct a model detacube on image plane using parameters and the grav. lensing.
    '''
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = params

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


def to_objectcoord_from(coord_image, PA, incl):
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
    # NOTE: convolving velocity is correct?
    model = convolve(velocity, index=0)

    return model


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


def initialize_globalparameters(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
) -> None:
    '''Set global parameters used in fitting.py.
    '''
    global imagecube, imagecube_error, xx_grid, yy_grid, vv_grid, lensing, convolve

    imagecube = np.copy(datacube.imageplane)
    imagecube_error = datacube.rms()
    imagecube_error = imagecube_error[:, np.newaxis, np.newaxis]
    vv_grid, yy_grid, xx_grid = datacube.coord_imageplane

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = no_convolve
    f_no_lensing: Callable = no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing


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
    initialize_globalparameters_moment(datacube, func_convolve, func_lensing, mom=0)
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
    initialize_globalparameters_moment(datacube, func_convolve, func_lensing, mom=1)
    func_fit = construct_model_moment1

    bound = (
        (0, 0, 0, -np.inf, 0, -np.inf, -np.inf),
        (0.5 * np.pi, np.inf, np.inf, np.inf, 2 * np.pi, np.inf, np.inf),
    )

    rms = datacube.rms_moment0()
    imagecube = datacube.pixmoment1(thresh=3 * rms)
    mask = np.isfinite(imagecube)

    args = (func_fit, mask)
    output = sp_least_squares(calculate_chi, init, args=args, bounds=bound)
    return output.x


def initialize_globalparameters_moment(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    mom: int = 0,
) -> None:
    '''Set global parameters used in fitting.py.
    '''
    global imagecube, imagecube_error, xx_grid, yy_grid, lensing, convolve

    if mom == 0:
        imagecube = datacube.moment0()
        imagecube_error = datacube.rms_moment0()
    elif mom == 1:
        rms = datacube.rms_moment0()
        imagecube = datacube.pixmoment1(thresh=3 * rms)
        idx = np.isfinite(imagecube)
        imagecube = imagecube[idx]  # imagecube becomes 1d
        mom0 = datacube.moment0()[idx]
        imagecube_error = 1 / np.sqrt(mom0)
    _, yy_grid, xx_grid = datacube.coord_imageplane
    xx_grid = xx_grid[0, :, :]
    yy_grid = yy_grid[0, :, :]

    # HACK: necessarily for mypy bug(?) https://github.com/python/mypy/issues/10740
    f_no_convolve: Callable = no_convolve
    f_no_lensing: Callable = no_lensing
    convolve = func_convolve if func_convolve else f_no_convolve
    lensing = func_lensing if func_lensing else f_no_lensing
