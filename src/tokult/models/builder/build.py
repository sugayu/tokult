'''Build model cube.
'''

from typing import NamedTuple, TypeVar, Generic, Optional, Callable
from collections import namedtuple

import numpy as np
import astropy.units as u

from .abstract import (
    AbstractCubeBuilder,
    AbstractBrightness,
    AbstractKinematics,
)
from .parameters import FitPar
from .brightness import ExponentialProfile
from .kinematics import FreemanDiskRotation
from .. import function
from ..utils import coordinates as coord
from ..utils import grid


##
def construct_model_at_imageplane_with(
    params: tuple[float, ...],
    xx_grid_image: np.ndarray,
    yy_grid_image: np.ndarray,
    vv_grid_image: np.ndarray,
    cubeshape_imageplane: tuple[int, ...],
    lensing: Optional[Callable] = None,
    create_interpolate_lensing: Optional[Callable] = None,
    upsampling_rate: tuple[int, ...] = (1, 1, 1),
) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.'''
    lensing = lensing if lensing else misc.no_lensing
    vv_grid = misc.gridding_upsample(vv_grid_image[:, 0, 0], upsampling_rate[0])
    vv_grid = vv_grid.reshape(-1, 1, 1)
    if (upsampling_rate[1] > 1) or (upsampling_rate[2] > 1):
        c.logger.warning(
            'Up-sampling in x and y has not yet been implemented. '
            'It performs fitting without up-sampling in x and y.'
        )
    yy_grid = yy_grid_image[0][np.newaxis, :, :]
    xx_grid = xx_grid_image[0][np.newaxis, :, :]
    xx_grid, yy_grid = lensing(xx_grid, yy_grid)
    lensing_interpolation = (
        create_interpolate_lensing(xx_grid_image, yy_grid_image)
        if create_interpolate_lensing
        else misc.no_lensing_interpolation
    )

    keys_globals = [
        'xx_grid',
        'yy_grid',
        'vv_grid',
        'lensing',
        'lensing_interpolation',
        'cubeshape_imageplane',
    ]
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


def construct_model_at_imageplane(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube on image plane using parameters and the grav. lensing.'''
    global xx_grid, yy_grid, vv_grid, cubeshape_imageplane
    _params = restore_params(params)
    p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = _params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    # velocity field
    coord_v = to_relativecoord_from(coordinate_abs, at_x0=p0, at_y0=p1)
    rr, pphi = to_objectcoord_from(coord_v, PA=p2, incl=p3)
    velocity = p5 + func.freeman_disk(rr, pphi, mass_dyn=10.0**p6, rnorm=p4, incl=p3)

    # spatial intensity distribution
    coord_i = to_relativecoord_from(coordinate_abs, at_x0=p10, at_y0=p11)
    rr_i, _ = to_objectcoord_from(coord_i, PA=p12, incl=p13)
    intensity = func.reciprocal_exp(rr_i, norm=p7, rnorm=p9)

    # create cube
    model = func.gaussian(vv_grid, center=velocity, sigma=p8, area=intensity)
    model = misc.down_sampling(model, cubeshape_imageplane)
    return model


def construct_model_moment0(params: list[float]) -> np.ndarray:
    '''Construct a model moment0 map convolved with dirtybeam.'''
    global xx_grid, yy_grid
    p0, p1, p2, p3, p4, p5 = params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    coord_i = to_relativecoord_from(coordinate_abs, at_x0=p0, at_y0=p1)
    rr_i, _ = to_objectcoord_from(coord_i, PA=p2, incl=p3)
    intensity = func.reciprocal_exp(rr_i, norm=p5, rnorm=p4)
    model = convolve(intensity)

    return model


def construct_model_moment1(params: list[float]) -> np.ndarray:
    '''Construct a model moment1 map convolved with dirtybeam.'''
    global xx_grid, yy_grid
    p0, p1, p2, p3, p4, p5, p6 = params

    coordinate_abs = np.moveaxis(np.array([xx_grid, yy_grid]), 0, -1)

    coord_v = to_relativecoord_from(coordinate_abs, at_x0=p5, at_y0=p6)
    rr, pphi = to_objectcoord_from(coord_v, PA=p4, incl=p0)
    velocity = p3 + func.freeman_disk(rr, pphi, mass_dyn=10.0**p2, rnorm=p1, incl=p0)
    # # NOTE: convolving velocity is correct?
    # model = convolve(velocity, index=0)

    return velocity
