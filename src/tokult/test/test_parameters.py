import numpy as np
import astropy.units as u
from ..parameters import FitPar
from .. import parameters as par
from ..mockobs import MockObservation
from ..fit.algorithms import EmceeMCMC
from ..utils.dataclass import fields


##
class ForTestParameters(par.FittingParametersBase):
    x: FitPar = FitPar(unit=u.pix, bound=(0, np.inf), initial=1.0)
    y: FitPar = FitPar(unit=u.pix, bound=(1.0, 10), initial=2.0)


def test_FittingParametersBase():
    p = ForTestParameters()
    assert isinstance(p.x, FitPar)
    assert p.x.initial == 1.0
    p_ntuple = p.namedtuplize((3.0, 2.0))
    assert p_ntuple.x == 3.0

    original = p.x.initial
    p.x.initial = 3.0 * original
    newp = ForTestParameters()
    assert newp.x.initial == original


def test_ParameterManager():
    pmanager = par.ParameterManager(mockobs=MockObservation(), optimizer=EmceeMCMC())
    assert pmanager.parameters['FreemanDiskParameters'].x0.fix is None

    mockobs = MockObservation()
    mockobs.models.galaxies.kinematic_model.name = 'disk0'
    mockobs.models.galaxies.kinematic_model.p.x0.fix = 5.0
    mockobs.models.galaxies.kinematic_model.p.PA.initial = 2.0
    pmanager = par.ParameterManager(mockobs=mockobs, optimizer=EmceeMCMC())
    assert np.isinf(pmanager.parameters['disk0'].x0.bound[0])
    assert pmanager.parameters['disk0'].PA.initial == 2.0

    longparam = tuple(np.arange(pmanager.nmax, dtype=float))
    shortparam = pmanager.shorten(longparam)
    longparam2 = pmanager.restore(shortparam)
    p_disk0 = pmanager.extract(longparam2, 'disk0')
    assert len(shortparam) == pmanager.nparams
    assert longparam[0] != longparam2[0]
    assert longparam2[0] == 5.0
    assert longparam[1:] == longparam2[1:]
    assert longparam[1:] == shortparam
    assert p_disk0[0] == 5.0
    assert len(p_disk0) == len(fields(mockobs.models.galaxies.kinematic_model.p))

    # initialvalues needs a input data cube.
    x, y, v = np.meshgrid(np.arange(11), np.arange(11), np.arange(11))
    data = (
        np.exp(-0.5 * (x - 5.0) ** 2)
        * np.exp(-0.5 * (y - 5.0) ** 2)
        * np.exp(-0.5 * (v - 5.0) ** 2)
    )
    init = pmanager.guessinitial(data)
    assert pmanager._initialvalues[0] == par.center_x
    assert init[0] == 5  # kin.x0
    assert pmanager._initialvalues[12] == par.max_brightness  # emi.brightness_center
    assert np.isclose(init[11], 1.0)  # 11 because kin.x0 is fiexed here

    init = pmanager.initialvalues(data)
    assert len(init) == len(shortparam)
    init = pmanager.initialvalues(data, ndim=3)
    assert init.shape == (3, len(shortparam))
    init = pmanager.initialvalues(data, seed=222, ndim=3)
    assert init[0, 0] != mockobs.models.galaxies.kinematic_model.p.y0.initial
