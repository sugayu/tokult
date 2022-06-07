'''Modules of fitting functions
'''
from __future__ import annotations
import numpy as np
from scipy.optimize import least_squares
import scipy.special as sps
from typing import Callable, Sequence, Optional, TYPE_CHECKING
from numpy.typing import ArrayLike
from .misc import rotate_coord, polar_coord

if TYPE_CHECKING:
    from .core import DataCube, DirtyBeam, GravLenz

##
'''Global parameters used for fitting

data: input DataCube contaning images.
xx_grid: coordinate indices of x for the image datacube.
yy_grid: coordinate indices of y for the image datacube.
vv_grid: coordinate indices of v for the image datacube.
pfs: DirtyBeam contaning dirtybeam (pfs) of the input datacube.
gl: GravLenz containing infomation to convert coordinates using gravitational lenzing.
'''
data: DataCube
xx_grid: np.ndarray
yy_grid: np.ndarray
vv_grid: np.ndarray
pfs: DirtyBeam
gl: GravLenz


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
    global data, xx_grid, yy_grid, vv_grid, pfs, gl
    data = data_in
    xx_grid, yy_grid, vv_grid = data_in.coord_imageplane
    if pfs_in:
        pfs = pfs_in
    if gl_in:
        gl = gl_in

    args_get_chi = (func_fit, data.imageplane, data.get_rms(), mask_use)
    output = least_squares(get_chi, init, args=args_get_chi, bounds=bound)

    p_bestfit = output.x
    dof = len(data.imageplane) - 1 - len(init)
    chi2 = np.sum(get_chi(p_bestfit, func_fit, data.imageplane, data.get_rms()) ** 2.0)
    J = output.jac
    # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
    cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
    result_error = np.sqrt(np.diag(cov))
    outputs = [p_bestfit, result_error, chi2, dof, cov]
    return Solution(outputs)


def get_chi(
    params: list[float],
    model_func: Callable,
    intensity: np.ndarray,
    intensity_err: np.ndarray,
    index: Optional[ArrayLike] = None,
) -> np.ndarray:
    model = model_func(params)
    if index:
        model[index] = model
    chi = (intensity - model) / intensity_err
    return chi


def func_freeman_disk(params: list[float]):
    '''Output Freeman disk.

    Global parameters
    ------------------
    data, xx_grid, yy_grid, vv_grid, pfs, gl
    '''
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = params

    # velocity field
    pos_grid = np.array([xx_grid - p0, yy_grid - p1]).transpose(1, 2, 3, 0)
    pos_grid = gl.lenz_image2source(pos_grid)
    pos_grid = rotate_coord(pos_grid, -p2)
    xx, yy = pos_grid.transpose(3, 0, 1, 2)
    yy = yy / np.cos(p3)  # inclination
    rr, pphi = polar_coord(xx, yy)
    r2h = 0.5 * rr / p4
    myu_0 = p6 / (2 * np.pi * p4 ** 2)
    G = 1
    A = sps.iv(0, r2h) * sps.kv(0, r2h) - sps.iv(1, r2h) * sps.kv(1, r2h)
    f_sightline = np.cos(pphi) * np.sin(p3)
    velocity = p5 + np.sqrt(4 * np.pi * G * myu_0 * p4 * r2h ** 2 * A) * f_sightline

    # spatial intensity distribution
    pos_grid = np.ndarray([xx_grid - p10, yy_grid - p11]).transpose(1, 2, 3, 0)
    pos_grid = gl.lenz_image2source(pos_grid)
    pos_grid = rotate_coord(pos_grid, -p12)
    xx_f, yy_f = pos_grid.transpose(3, 0, 1, 2)
    yy_f = yy_f / np.cos(p13)
    rr_f, _ = polar_coord(xx_f, yy_f)
    intensity = p7 * np.exp(-rr_f / p9)

    # convolvolution
    model = func_Gauss(vv_grid, center=velocity, sigma=p8, area=intensity)
    model_convolved = np.zeros(xx_grid.shape)
    for i in range(xx_grid.shape[2]):
        model_convolved[:, :, i] = pfs.convolve(model[:, :, i])

    # # Choose fitting region via indices.
    # iv = (v - dv * c.conf.chan_start - vel_start) / dv
    # iv = np.round(iv).astype(int)
    # ix = (x - xlow).astype(int)
    # iy = (y - ylow).astype(int)
    # Imodel_last = model[iy, ix, iv]

    return model_convolved


def func_Gauss(x, center, sigma, area):
    '''Gaussian function.
    '''
    norm = area / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


class Solution:
    '''Contains output solutions of fittings.
    '''

    def __init__(self, outputs_fitting):
        pass
