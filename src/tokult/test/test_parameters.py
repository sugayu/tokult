import numpy as np
import astropy.units as u
from ..parameters import ParameterManager, FittingParametersBase, FitPar
from ..mockobs import MockObservation
from ..fit.algorithms import EmceeMCMC
from ..utils.dataclass import fields


##
class TestParameters(FittingParametersBase):
    x: FitPar = FitPar(unit=u.pix, bound=(0, np.inf), initial=1.0)


def test_FittingParametersBase():
    p = TestParameters()
    assert isinstance(p.x, FitPar)
    assert p.x.initial == 1.0
    p_ntuple = p.namedtuplize((3.0,))
    assert p_ntuple.x == 3.0

    original = p.x.initial
    p.x.initial = 3.0 * original
    newp = TestParameters()
    assert newp.x.initial == original


def test_ParameterManager():
    pmanager = ParameterManager(mockobs=MockObservation(), optimizer=EmceeMCMC())
    assert pmanager.parameters['FreemanDiskParameters'].x0.fix is None

    mockobs = MockObservation()
    mockobs.models.galaxies.kinematic_model.name = 'disk0'
    mockobs.models.galaxies.kinematic_model.p.x0.fix = 5.0
    pmanager = ParameterManager(mockobs=mockobs, optimizer=EmceeMCMC())
    assert np.isinf(pmanager.parameters['disk0'].x0.bound[0])

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

    init = pmanager.initialvalues()
    assert len(init) == len(shortparam)
    init = pmanager.initialvalues(ndim=3)
    assert init.shape == (3, len(shortparam))
