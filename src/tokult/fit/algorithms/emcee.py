'''MCMC with emcee
'''

from __future__ import annotations
from typing import Callable, Sequence
import numpy as np
import emcee
from emcee.moves import DEMove, DESnookerMove
from ..optimize import Optimizer
from ..solution import Solution


__all__ = ['EmceeMCMC', 'McmcSolution']


##
class McmcSolution(Solution):
    ''' '''

    def __init__(self, sampler) -> None: ...

    # @classmethod
    # def from_mcmc(cls, params: np.ndarray, chi2: float, dof: float) -> Solution:
    #     '''Construct Solution() from MCMC.'''


class EmceeMCMC(Optimizer):
    '''MCMC optimizer using emcee.'''

    def __init__(self) -> None:
        super().__init__()
        self.nwalkers: int
        self.ndim: int
        self.nsteps: int
        self.moves: list

    def optimize(self) -> McmcSolution:
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.calculate_probability,
            # args=args,
            # pool=pool,
            moves=self.moves,
        )
        init = self.fullparams.initialparam
        sampler.run_mcmc(init, self.nsteps)
        return McmcSolution(sampler)

    def configure(
        self, nwalkers: int, ndim: int, nsteps: int, moves: list | None = None
    ) -> None:
        self.nwalkers = nwalkers
        self.ndim = ndim
        self.nsteps = nsteps
        if moves is None:
            self.moves = [(DEMove(), 0.8), (DESnookerMove(), 0.2)]
        else:
            self.moves = moves

    def calculate_probability(self, params: tuple[float, ...]) -> float:
        '''Calcurate log probability.'''
        log_prior = self.calculate_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        log_likelihood = self.calculate_likelihood(params)
        return log_prior + log_likelihood

    def calculate_prior(self, params: tuple[float, ...]) -> float:
        '''Calcurate log prior of parameters.'''
        # _params = np.array(params)
        # bound0, bound1 = (np.array(bound[0]), np.array(bound[1]))
        # if np.all(bound0 < _params) and np.all(_params < bound1):
        return 0.0
        # return -np.inf

    def calculate_likelihood(self, params: tuple[float, ...]) -> float:
        '''Calcurate chi = (data-model)/error.'''
        model = self.modeling(params)
        if self.data.mask is not None:
            model = model[self.data.mask]
        chi = (self.data.data - model) / self.data.uncertainty
        return -0.5 * np.sum(abs(chi.ravel()) ** 2)


#     def optimize(
#         config: c.ConfigParameters,
#         datacube: DataCube,
#         init: Sequence[float],
#         bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
#         fix: Optional[FixParams] = None,
#         func_convolve: Optional[Callable] = None,
#         func_lensing: Optional[Callable] = None,
#         func_create_lensinginterp: Optional[Callable] = None,
#         beam_vis: Optional[np.ndarray] = None,
#         norm_weight: Optional[float] = None,
#         mask_for_fit: Optional[np.ndarray] = None,
#         mode_fit: str = 'image',
#         is_separate: bool = False,
#         nwalkers: int = 64,
#         nsteps: int = 5000,
#         pool: Optional[Pool] = None,
#         progressbar: bool = False,
#     ) -> Solution:
#         '''MCMC using emcee'''
#         rng = default_rng(222)

#         if mode_fit == 'image':
#             if mask_for_fit is None:
#                 mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)
#             initialize_globalparameters_for_image(
#                 datacube,
#                 mask_for_fit,
#                 func_convolve,
#                 func_lensing,
#                 func_create_lensinginterp,
#                 config.noisescale_factor,
#                 config.pixel_upsampling_rate,
#             )
#             func_fit = construct_convolvedmodel
#         elif mode_fit == 'uv':
#             if beam_vis is None:
#                 raise ValueError('Parameter "beam_vis" is necessary for uvfit.')
#             if norm_weight is None:
#                 raise ValueError('Parameter "norm_weight" is necessary for uvfit.')
#             if mask_for_fit is None:
#                 mask_for_fit = np.ones_like(datacube.uvplane).astype(bool)

#             initialize_globalparameters_for_uv(
#                 datacube,
#                 beam_vis,
#                 norm_weight,
#                 mask_for_fit,
#                 func_lensing,
#                 func_create_lensinginterp,
#                 config.noisescale_factor,
#                 config.pixel_upsampling_rate,
#             )
#             func_fit = construct_uvmodel
#         else:
#             raise ValueError(
#                 f'mode_fit is "image" or "uv", no option for "{mode_fit}".'
#             )

#         set_fixedparameters(fix, is_separate)
#         bound = get_bound_params() if bound is None else bound
#         _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
#         if is_init_outside_of_bound(_init, _bound):
#             raise ValueError('The "init" is outside of the "bound".')
#         args = (func_fit, _bound)

#         ndim = len(_init)
#         __init = np.array(_init)
#         norm = rng.standard_normal((nwalkers, ndim))
#         __init = __init + __init * config.mcmc_init_dispersion * norm

#         if pool is not None:
#             map_globals_to_childprocesses(pool)

#         sampler = emcee.EnsembleSampler(
#             nwalkers,
#             ndim,
#             calculate_log_probability,
#             args=args,
#             pool=pool,
#             moves=config.mcmc_moves,
#         )
#         sampler.run_mcmc(__init, nsteps, progress=progressbar)

#         dof = datacube.imageplane.size - 1 - len(_init)
#         return Solution.from_sampler(sampler, dof, func_fit)
