import numpy as np
from astropy.nddata import NDData
from ..ui import Tokult
from ..fit.algorithms import EmceeMCMC


##
def test_Toult():
    tok = Tokult(None)
    assert isinstance(tok.data, NDData)

    v, y, x = np.meshgrid(np.arange(30), np.arange(100), np.arange(100), indexing='ij')
    data = (
        3.0
        * np.exp(-((x - 50.0) ** 2 / (2.0 * 10**2)))
        * np.exp(-((y - 40.0) ** 2 / (2.0 * 2.0**2)))
        * np.exp(-((v - 15.0) ** 2 / (2.0 * 1.0**2)))
    )
    tok.data = NDData(data, uncertainty=np.ones((30, 100, 100)))
    tok.optimizer = EmceeMCMC(nwalkers=28, nsteps=3)
    sol = tok.runfit()
