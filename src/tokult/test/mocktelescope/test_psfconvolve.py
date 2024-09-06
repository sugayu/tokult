import numpy as np
from ...mocktelescope.psfconvolve import PointSpreadFunction


def test_PointSpreadFunction():
    image = np.zeros((8, 8), dtype=float)
    image[4, 4] = 1.0
    psfimage = np.zeros((8, 8), dtype=float)
    psfimage[3:6, 3:6] = np.array(
        [[0.05, 0.1, 0.05], [0.1, 0.4, 0.1], [0.05, 0.1, 0.05]]
    )

    psf = PointSpreadFunction(psfimage)
    assert np.all(np.isclose(psf.convolve(image), psfimage))
