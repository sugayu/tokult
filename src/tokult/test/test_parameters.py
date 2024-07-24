import numpy as np
import astropy.units as u
from ..parameters import ParameterManager, FittingParametersBase, FitPar
from ..mockobs import MockObservation
from ..fit.algorithms import EmceeMCMC


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
    pmanager = ParameterManager(mockobs=mockobs, optimizer=EmceeMCMC())
    assert np.isinf(pmanager.parameters['disk0'].x0.bound[0])
