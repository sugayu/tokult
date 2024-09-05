import numpy as np
from ....fit.algorithms import EmceeMCMC
from ....parameters import ParameterManager


class MockParameterManager:
    bounds = np.array(([0.0, 1.0, 2.0], [10.0, 20.0, 30.0]))

    within_boundaries = ParameterManager.within_boundaries


def test_calcurate_prior():
    mcmc = EmceeMCMC()
    mcmc.pmanager = MockParameterManager()
    assert mcmc.calculate_prior((1.0, 2.0, 10.0)) == 0.0
    assert np.isinf(mcmc.calculate_prior((1.0, 2.0, 0.0)))
