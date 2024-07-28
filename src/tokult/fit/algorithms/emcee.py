'''MCMC with emcee
'''

from __future__ import annotations
from logging import getLogger
import numpy as np
import emcee
from emcee.moves import DEMove, DESnookerMove
from ..optimize import Optimizer
from ..solution import Solution


__all__ = ['EmceeMCMC', 'MCMCSolution']

logger = getLogger(__name__)


##
class MCMCSolution(Solution):
    ''' '''

    def __init__(self, sampler) -> None:
        flat = sampler.get_chain(discard=300, thin=4, flat=True)
        logger.info(np.mean(flat, axis=0))

    # @classmethod
    # def from_mcmc(cls, params: np.ndarray, chi2: float, dof: float) -> Solution:
    #     '''Construct Solution() from MCMC.'''


class EmceeMCMC(Optimizer):
    '''MCMC optimizer using emcee.'''

    def __init__(
        self,
        *,
        nwalkers: int = 64,
        nsteps: int = 500,
        moves: list = [(DEMove(), 0.8), (DESnookerMove(), 0.2)],
    ) -> None:
        super().__init__()
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.moves = moves

    def optimize(self, initial: np.ndarray | None) -> MCMCSolution:
        sampler = emcee.EnsembleSampler(
            self.nwalkers,
            self.ndim,
            self.calculate_probability,
            # args=args,
            # pool=pool,
            moves=self.moves,
        )
        # it's a big confusing, but ndim=self.nwalkers is correct.
        init = self.pmanager.initialvalues(
            initial=initial, seed=222, ndim=self.nwalkers
        )
        # initial check
        for i in init:
            self.pmanager.warn_if_outside_boundaries(i)

        sampler.run_mcmc(init, self.nsteps)
        return MCMCSolution(sampler)

    def calculate_probability(self, params: tuple[float, ...]) -> float:
        '''Calcurate log probability.'''
        log_prior = self.calculate_prior(params)
        if not np.isfinite(log_prior):
            return -np.inf
        log_likelihood = self.calculate_likelihood(params)
        return log_prior + log_likelihood

    def calculate_prior(self, params: tuple[float, ...]) -> float:
        '''Calcurate log prior of parameters.'''
        if self.pmanager.within_boundaries(params):
            return 0.0
        return -np.inf

    def calculate_likelihood(self, params: tuple[float, ...]) -> float:
        '''Calcurate chi = (data-model)/error.'''
        model = self.modeling(params)
        if self.data.mask is not None:
            model = model[self.data.mask]
        chi = (self.data.data - model) / self.data.uncertainty.array
        r = -0.5 * np.sum(abs(chi.ravel()) ** 2)
        return r

    @property
    def ndim(self) -> int:
        return self.pmanager.nparams


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
