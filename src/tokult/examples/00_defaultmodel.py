'''Simple example using default thindisk models.
'''

import numpy as np
from numpy.random import default_rng
from astropy.nddata import NDData
import tokult
from tokult.fit.algorithms import EmceeMCMC
from tokult import visualization as vis
from sugayutils.figure import makefig
from sugayutils.log import mylogconfig

mylogconfig(level='INFO')


##
def _main():
    tok = tokult.Tokult(None)
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
    model = tok.build_model(param)
    for i in range(10, 20):
        fig = makefig(figsize=[3.5, 3.5])
        ax = fig.add_subplot(1, 1, 1)
        ax.imshow(model[i], origin='lower')
        fig.save_or_plot()


def main():
    tok = tokult.Tokult(data=NDData(np.empty((30, 100, 100))))
    # fmt:off
    param = (50.0, 50.0, np.pi / 2, np.pi / 3, 10.0, 15.0, 4.0,
             50.0, 50.0, np.pi / 2, np.pi / 3, 10.0, 8.0, 5.0)
    # fmt:on
    datamodel = tok.build_model(param)
    rng = default_rng(222)
    noise = rng.standard_normal((30, 100, 100)) * 0.05
    tok.data = NDData(datamodel + noise, uncertainty=np.ones((30, 100, 100)) * 0.05)

    kin = tokult.models.kinematics.FreemanDiskRotation()
    kin.name = 'kinematics'

    emi = tokult.models.brightness.ExponentialProfile()
    emi.name = 'emission'
    emi.p.x0.fix = 'kinematics.x0'
    emi.p.y0.fix = 'kinematics.y0'
    emi.p.PA.fix = 'kinematics.PA'
    emi.p.inclination.fix = 'kinematics.inclination'
    emi.p.radius.fix = 'kinematics.radius'

    tok.observation.models.galaxies.kinematic_model = kin
    tok.observation.models.galaxies.brightness_model = emi
    tok.optimizer = EmceeMCMC(nwalkers=28, nsteps=5000, progress=True)

    sol = tok.runfit()

    chain = sol.sampler.get_chain(thin=50)
    best = np.mean(sol.sampler.get_chain(discard=1000, thin=50, flat=True), axis=0)
    bestmodel = tok.build_model(best)

    vis.show_residuals(datamodel + noise, bestmodel)

    return sol


if __name__ == '__main__':
    sol = main()
