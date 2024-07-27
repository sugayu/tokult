import numpy as np
from ....models.brightness import ExponentialProfile
from sugayutils.figure import makefig


def test_ExponentialProfile():
    exp = ExponentialProfile()
    coord = np.transpose(np.meshgrid(np.arange(10), np.arange(10)), (2, 1, 0))
    exp.coord_yx = coord
    model = exp((5.0, 3.0, 0.0, np.pi / 3.0, 2.0, 1.0))
    # fig = makefig(figsize=[3.5, 3.5])
    # ax = fig.add_subplot(1, 1, 1)
    # ax.imshow(model, origin='lower')
    # fig.save_or_plot()
    assert model[3, 5] == 1.0
    assert model[0, 0] > 0.0
