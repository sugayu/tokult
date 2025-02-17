import pickle
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from numpy.random import default_rng
from astropy.nddata import NDData
from ...data import sharedmemory


##
def test_SharedMemoryNDData():
    shape = (10, 10)
    rng = default_rng()
    nddata = NDData(rng.standard_normal(shape), uncertainty=np.ones(shape))

    with SharedMemoryManager() as smm:
        smdata = sharedmemory.SharedMemoryNDData(nddata, smmanager=smm)
        smdata2 = pickle.loads(pickle.dumps(smdata))
        assert np.all(np.equal(smdata.data, smdata2.data))
