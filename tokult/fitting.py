'''Modules of fitting functions
'''
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
from typing import Callable, Sequence, Optional, TYPE_CHECKING
from numpy.typing import ArrayLike
from . import function as func
from .misc import rotate_coord, polar_coord, no_lenzing, no_convolve

if TYPE_CHECKING:
    from .core import DataCube, DirtyBeam, GravLenz

##
'''Global parameters used for fitting

data: input DataCube contaning images.
xx_grid: coordinate indices of x for the image datacube.
yy_grid: coordinate indices of y for the image datacube.
vv_grid: coordinate indices of v for the image datacube.
convolve: a function to convolve the input datacube. Method of DirtyBeam.
lenzing: a function to convert source plane to image plane. Method of GravLenz.
'''
data: DataCube
xx_grid: np.ndarray
yy_grid: np.ndarray
vv_grid: np.ndarray
lenzing: Callable = no_lenzing
convolve: Callable = no_convolve


def least_square(
    data_in: DataCube,
    init: Sequence[float],
    bound: Sequence[float],
    func_fit: Callable,
    pfs_in: Optional[DirtyBeam] = None,
    gl_in: Optional[GravLenz] = None,
    mask_use: Optional[ArrayLike] = None,
) -> Solution:
    '''Least square fitting using scipy.optimize.least_squares
    '''
    global data, xx_grid, yy_grid, vv_grid, lenzing, convolve
    data = data_in
    xx_grid, yy_grid, vv_grid = data_in.coord_imageplane
    if pfs_in:
        convolve = pfs_in.convolve
    if gl_in:
        lenzing = gl_in.lenzing

    args = (func_fit, data.imageplane, data.get_rms(), mask_use)
    output = least_squares(calculate_chi, init, args=args, bounds=bound)

    p_bestfit = output.x
    dof = len(data.imageplane) - 1 - len(init)
    chi2 = np.sum(
        calculate_chi(p_bestfit, func_fit, data.imageplane, data.get_rms()) ** 2.0
    )
    J = output.jac
    # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
    cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
    result_error = np.sqrt(np.diag(cov))
    outputs = [p_bestfit, result_error, chi2, dof, cov]
    return Solution(outputs)


def calculate_chi(
    params: list[float],
    model_func: Callable,
    intensity: np.ndarray,
    intensity_error: np.ndarray,
    index: Optional[ArrayLike] = None,
) -> np.ndarray:
    '''Calcurate chi = (data-model)/error for least square fitting
    '''
    model = model_func(params)
    if index:
        model = model[index]

        # # Choose fitting region via indices.
        # iv = (v - dv * c.conf.chan_start - vel_start) / dv
        # iv = np.round(iv).astype(int)
        # ix = (x - xlow).astype(int)
        # iy = (y - ylow).astype(int)
        # model_last = model[iy, ix, iv]

    chi = (intensity - model) / intensity_error

    return chi


def create_model_convolved(params: list[float]):
    '''Create a model detacube convolved with dirtybeam.
    '''
    model = create_model_at_imageplane(params)

    model_convolved = np.zeros(xx_grid.shape)
    for i in range(xx_grid.shape[2]):
        model_convolved[:, :, i] = convolve(model[:, :, i])

    return model_convolved


def create_model_at_imageplane(params: list[float]):
    '''Create a model detacube on image plane using parameters and the grav. lenzing.
    '''
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = params

    # velocity field
    coord_image_v = np.array([xx_grid - p0, yy_grid - p1]).transpose(1, 2, 3, 0)
    rr, pphi = to_objectcoord_from(coord_image_v, PA=p2, incl=p3)
    velocity = p5 + func.freeman_disk(rr, pphi, mass_dyn=p6, rnorm=p4, incl=p3)

    # spatial intensity distribution
    coord_image_i = np.array([xx_grid - p10, yy_grid - p11]).transpose(1, 2, 3, 0)
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

    coord_source = lenzing(coord_image)
    coord_object = rotate_coord(coord_source, pa)
    xx, yy = coord_object.transpose(3, 0, 1, 2)
    yy = yy / np.cos(inclination)
    r, phi = polar_coord(xx, yy)
    return r, phi


class Solution:
    '''Contains output solutions of fittings.
    '''

    def __init__(self, outputs_fitting):
        pass
