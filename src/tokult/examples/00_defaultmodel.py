'''Simple example using default thindisk models.
'''

import numpy as np
from astropy.nddata import NDData
from tokult import Tokult
from tokult.fit.algorithms import EmceeMCMC
from sugayutils.figure import makefig
from sugayutils.log import mylogconfig

mylogconfig(level='DEBUG')


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
    model = tok.model(
        (
            50.0,
            50.0,
            0.0,
            np.pi / 3,
            10.0,
            15.0,
            4.0,
            50.0,
            50.0,
            0.0,
            np.pi / 3,
            10.0,
            10.0,
            5.0,
        )
    )
    for i in range(10, 20):
        fig = makefig(figsize=[3.5, 3.5])
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(model[i], origin='lower')
        fig.save_or_plot()


def main():
    tok = Tokult(data=NDData(np.empty((30, 100, 100))))
    param = (
        50.0,
        50.0,
        0.0,
        np.pi / 3,
        10.0,
        15.0,
        4.0,
        50.0,
        50.0,
        0.0,
        np.pi / 3,
        10.0,
        10.0,
        5.0,
    )
    model = tok.model(param)
    tok.data = NDData(model, uncertainty=np.ones((30, 100, 100)) * 0.001)
    tok.optimizer = EmceeMCMC(nwalkers=28, nsteps=3000)
    sol = tok.runfit(initial=param)


if __name__ == '__main__':
    main()
