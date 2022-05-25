import numpy as np
from tokult import casa


def test_glenzing():
    glenzing = casa.glenzing
    pos = np.arange(18).reshape(9, 2)
    g1, g2, k = np.arange(9), np.arange(9), np.arange(9)
    ans = glenzing(pos, g1, g2, k)
    assert ans.shape == pos.shape


def test_rotate_coord():
    rotate_coord = casa.rotate_coord
    pos = np.array([np.arange(3), np.arange(3)]).T.astype(float)
    assert np.allclose(rotate_coord(pos, 0.0), pos)
    assert np.allclose(rotate_coord(pos, np.pi / 2.0), pos * np.array([-1, 1]))
    assert np.allclose(rotate_coord(pos, np.pi), pos * np.array([-1, -1]))
