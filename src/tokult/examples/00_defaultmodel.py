'''Simple example using default thindisk models.
'''

import numpy as np
from astropy.nddata import NDData
from tokult import Tokult
from tokult.fit.algorithms import EmceeMCMC
from tokult import visualization as vis
from sugayutils.figure import makefig
from sugayutils.log import mylogconfig

mylogconfig(level='INFO')


##
def _main():
    tok = Tokult(None)
    assert isinstance(tok.data, NDData)

    v, y, x = np.meshgrid(np.arange(30), np.arange(100), np.arange(100), indexing='ij')
    data = (
        3.0
        * np.exp(-((x - 50.0) ** 2 / (2.0 * 10**2)))
        * np.exp(-((y - 40.0) ** 2 / (2.0 * 2.0**2)))
        * np.exp(-((v - 15.0) ** 2 / (2.0 * 4.0**2)))
    )
    tok.data = NDData(data, uncertainty=np.ones((30, 100, 100)) * 0.001)
    # fmt:off
    param = (50.0, 50.0, 0.0, np.pi / 3, 10.0, 15.0, 4.0,
             50.0, 50.0, 0.0, np.pi / 3, 10.0, 10.0, 5.0)
    # fmt:on
    model = tok.model(param)
    for i in range(10, 20):
        fig = makefig(figsize=[3.5, 3.5])
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(model[i], origin='lower')
        fig.save_or_plot()


def main():
    tok = Tokult(data=NDData(np.empty((30, 100, 100))))
    # fmt:off
    param = (50.0, 50.0, 0.0, np.pi / 3, 10.0, 15.0, 4.0,
             50.0, 50.0, 0.0, np.pi / 3, 10.0, 10.0, 5.0)
    datamodel = tok.model(param)
    tok.data = NDData(datamodel, uncertainty=np.ones((30, 100, 100)) * 0.001)
    tok.optimizer = EmceeMCMC(nwalkers=28, nsteps=5000, progress=True)
    param = (47.0, 52.0, 0.1, np.pi / 4, 12.0, 13.0, 5.0,
             52.0, 49.0, 0.01, np.pi / 2.5, 9.0, 11.0, 4.9)
    # fmt:on
    sol = tok.runfit(initial=param)

    best = np.mean(sol.sampler.get_chain(discard=2000, thin=50, flat=True), axis=0)
    bestmodel = tok.model(best)

    vis.show_residuals(datamodel, bestmodel)

    return sol


if __name__ == '__main__':
    sol = main()
