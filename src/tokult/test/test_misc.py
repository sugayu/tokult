import numpy as np
from tokult import misc


def test_down_sampling():
    a = np.arange(4 * 8 * 10).reshape(4, 8, 10)
    b = misc.down_sampling(a, (2, 2, 5))
    assert b.shape == (2, 2, 5)
    assert np.isclose(np.mean(a), np.mean(b))
    assert np.array_equal(misc.down_sampling(a, (4, 8, 10)), a, equal_nan=True)


def test_gridding_upsampling():
    a = np.array([0, 1, 2]).astype(float)
    assert np.all(misc.gridding_upsample(a, 1) == a)
    assert np.all(
        misc.gridding_upsample(a, 2) == np.array([-0.25, 0.25, 0.75, 1.25, 1.75, 2.25])
    )
