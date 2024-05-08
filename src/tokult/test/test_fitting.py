import numpy as np
from tokult import fitting


def test__correct_cube_error_for_uv():
    a = np.arange(3 * 6 * 4).reshape(3, 6, 4).astype(float)
    b = fitting._correct_cube_error_for_uv(a)
    assert a[0, 3, 0] * np.sqrt(2) == b[0, 3, 0]


def test__add_mask_for_uv():
    a = np.ones((3, 6, 4)).astype(bool)
    b = fitting._add_mask_for_uv(a)
    assert np.all(np.logical_not(b[:, 1, 0]))
    assert np.all(np.logical_not(b[:, 2, -1]))
    assert np.all(b[:, 0, -1])
