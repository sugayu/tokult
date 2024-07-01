'''Least square fitting.
'''
from .optimize import Optimizer, SolutionDI


##
class MCMC(Optimizer):
    ''' '''

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
        '''MCMC using emcee'''
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
                config.noisescale_factor,
                config.pixel_upsampling_rate,
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
                config.noisescale_factor,
                config.pixel_upsampling_rate,
            )
            func_fit = construct_uvmodel
        else:
            raise ValueError(
                f'mode_fit is "image" or "uv", no option for "{mode_fit}".'
            )

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


class MCMCDI(SolutionDI):
    ''' '''

    def __init__(self) -> None:
        ...

    @classmethod
    def from_montecarlo(cls, params: np.ndarray, chi2: float, dof: float) -> Solution:
        '''Construct Solution() from montecarlo perturbations.'''
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
