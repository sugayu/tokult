import numpy as np
from astropy.nddata import NDData
from ..ui import Tokult


##
def test_Toult():
    tok = Tokult(None)
    assert isinstance(tok.data, NDData)

    tok.data = NDData(np.zeros((30, 100, 100)), uncertainty=np.ones((30, 100, 100)))
    sol = tok.runfit()
