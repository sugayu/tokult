from astropy.nddata import NDData
from ..ui import Tokult


##
def test_Toult():
    tok = Tokult(None)
    assert isinstance(tok.data, NDData)
