import pickle
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from numpy.random import default_rng
from astropy.nddata import NDData
from ...data import sharedmemory


##
def test_SharedMemoryNDData():
    shape = (100, 100)
    rng = default_rng()
    nddata = NDData(rng.standard_normal(shape), uncertainty=np.ones(shape))

    with SharedMemoryManager() as smm:
        smdata = sharedmemory.SharedMemoryNDData(nddata, smmanager=smm)
        smdata2 = pickle.loads(pickle.dumps(smdata))
        smdata.data
        smdata.uncertainty
        assert np.all(np.equal(nddata.data, smdata.data))
        assert np.all(np.equal(smdata.data, smdata2.data))
        assert pickle.dumps(smdata).__sizeof__() < 1000
