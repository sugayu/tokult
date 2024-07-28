'''Estimate uncertainty by montecarlo resampling.
'''

from ..optimize import Optimizer
from ..solution import Solution

__all__ = ['MonteCarlo']


##
class MonteCarlo(Optimizer):
    ''' '''


#     def __init__(self) -> None: ...

#     def montecarlo(
#         config: c.ConfigParameters,
#         datacube: DataCube,
#         init: Sequence[float],
#         bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
#         fix: Optional[FixParams] = None,
#         func_convolve: Optional[Callable] = None,
#         func_fullconvolve: Optional[Callable] = None,
#         func_lensing: Optional[Callable] = None,
#         func_create_lensinginterp: Optional[Callable] = None,
#         mask_for_fit: Optional[np.ndarray] = None,
#         uvcoverage: Optional[np.ndarray] = None,
#         nperturb: int = 1000,
#         niter: int = 1,
#         is_separate: bool = False,
#         progressbar: bool = False,
#     ) -> Solution:
#         '''Monte Carlo fitting to derive errors using scipy.optimize.least_squares'''

#         if mask_for_fit is None:
#             mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)

#         initialize_globalparameters_for_image(
#             datacube,
#             mask_for_fit,
#             func_convolve,
#             func_lensing,
#             func_create_lensinginterp,
#             config.noisescale_factor,
#             config.pixel_upsampling_rate,
#         )
#         func_fit = construct_convolvedmodel

#         set_fixedparameters(fix, is_separate)
#         bound = get_bound_params() if bound is None else bound
#         _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
#         if is_init_outside_of_bound(_init, _bound):
#             raise ValueError('The "init" is outside of the "bound".')
#         args = (func_fit,)

#         params_mc = np.empty((nperturb, len(_init)))
#         rms_of_standardnoise = datacube._estimate_rms_of_standardnoise(
#             shape=datacube.original.shape, convolve=func_fullconvolve
#         )
#         for j in tqdm.tqdm(range(nperturb), leave=None, disable=(not progressbar)):
#             _init_j = _init
#             global cube, mask
#             noisycube_originalsize = datacube.perturbed(
#                 convolve=func_fullconvolve,
#                 rms_of_standardnoise=rms_of_standardnoise,
#                 uvcoverage=uvcoverage,
#                 is_originalsize=True,
#             )
#             noisycube = noisycube_originalsize[
#                 datacube.vslice, datacube.yslice, datacube.xslice
#             ]
#             cube = noisycube[mask]
#             for _ in range(niter):
#                 output = sp_least_squares(
#                     calculate_chi, _init_j, args=args, bounds=_bound
#                 )
#                 _init_j = output.x
#             params_mc[j, :] = output.x

#             config._debug.mc_savecube(noisycube, noisycube_originalsize, j)

#         dof = datacube.imageplane.size - 1 - len(_init)
#         chi2 = np.sum(calculate_chi(output.x, func_fit) ** 2.0)
#         return Solution.from_montecarlo(params_mc, chi2, dof)


# class MonteCarloDI(SolutionDI):
#     ''' '''

#     def __init__(self) -> None: ...

#     def from_sampler(
#         cls,
#         sampler: emcee.EnsembleSampler,
#         dof: float,
#         func_fit: Callable,
#         # mask_for_fit: Optional[np.ndarray] = None,
#     ) -> Solution:
#         '''Construct Solution() from sampler.'''
#         try:
#             tau = sampler.get_autocorr_time()
#             burnin = int(np.max(tau) * 2.0)
#             thin = int(np.min(tau) / 2.0)
#         except emcee.autocorr.AutocorrError:
#             logger.warning(
#                 'MCMC may not be converged.'
#                 'Please be careful to use the best-fit parameters.'
#             )
#             # HACK: these estimates may be wrong.
#             shape = sampler.get_chain(discard=0, thin=1).shape
#             burnin = int(shape[0] / 100.0 * 2.0)
#             thin = int(shape[0] / 100.0 / 2.0)
#         flat_samples = sampler.get_chain(discard=burnin, thin=thin, flat=True)

#         p16, p50, p84 = np.percentile(flat_samples, [16, 50, 84], axis=0)
#         best = restore_params(p50)
#         error_high = restore_params(p84 - p50)
#         error_low = restore_params(p50 - p16)

#         chi2 = np.sum(calculate_chi(best, func_fit) ** 2.0)
#         return cls(
#             best,
#             error_high,
#             error_low,
#             chi2,
#             dof,
#             np.array(0.0),
#             mode_fitting='mcmc',
#             sampler=sampler,
#         )
