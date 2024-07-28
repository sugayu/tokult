import numpy as np
from ....models.utils import coordinates as coord


def test_rotate():
    pos = np.array([[[0.0]], [[1.0]]])
    pos = np.transpose(pos, (1, 2, 0))
    output = np.transpose(coord.rotate(pos, np.pi / 2), (2, 0, 1))
    assert np.all(np.isclose(output, np.array([[[1.0]], [[0.0]]])))

    pos0 = np.array(np.meshgrid(np.arange(3), np.arange(3), indexing='ij'))
    pos = np.transpose(pos0, (1, 2, 0))
    assert np.all(np.isclose(coord.rotate(pos, angle=0.0), pos))
    result = np.transpose(pos0, (0, 2, 1)).copy()
    result[1] = result[1] * -1.0
    result = np.transpose(result, (1, 2, 0))
    assert np.all(np.isclose(coord.rotate(pos, angle=np.pi / 2.0), result))
