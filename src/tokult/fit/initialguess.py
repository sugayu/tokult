'''Guess initial fitting parameters.
'''

##
def initialguess(
    datacube: DataCube,
    func_convolve: Optional[Callable] = None,
    func_lensing: Optional[Callable] = None,
    func_create_lensinginterp: Optional[Callable] = None,
    is_separate: bool = False,
) -> InputParams:
    '''Guess initial parameters by fitting moment 0 and 1 maps.'''
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
        datacube, mask, func_convolve, func_lensing, func_create_lensinginterp, mom=1
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
    '''Set global parameters used in fitting.py.'''
    global cube, cube_error, xx_grid, yy_grid
    global lensing, lensing_interpolation, convolve, mask

    mask = mask_for_fit
    if mom == 0:
        cube = datacube.moment0()[mask.squeeze()]
        cube_error = np.array(datacube.rms_moment0())
    elif mom == 1:
        rms = datacube.rms_moment0()
        cube = datacube.pixmoment1(thresh=3 * rms)
        idx = np.isfinite(cube) and mask.squeeze()
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
