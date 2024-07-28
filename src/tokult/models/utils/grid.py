'''Functions related with coordinate grids.
'''

import numpy as np


##
def down_sampling(cube: np.ndarray, shape_to: tuple[int, ...]) -> np.ndarray:
    '''Down-sampling of a data cube.

    This is to reconstruct the data cube more-finely-resampled when the sampling
    rate is not sufficient.

    Args:
        cube (np.ndarray): 3d data cube.
        shape_to (tuple[int, int, int]): Shape of the resampled cube.

    Returns:
        np.ndarray: Resampled data cube.
    '''
    shape_from = cube.shape
    if shape_from == shape_to:
        return cube
    nbins = [f // t for f, t in zip(shape_from, shape_to)]
    cube_out = cube.reshape(
        shape_to[0], nbins[0], shape_to[1], nbins[1], shape_to[2], nbins[2]
    )
    return cube_out.mean(axis=(1, 3, 5))


def gridding_upsample(grid: np.ndarray, rate_upsampling: int) -> np.ndarray:
    '''Make grids up-sampling.

    Args:
        grid (np.ndarray):
        rate_upsampling (int):
    '''
    if rate_upsampling == 1:
        return grid
    nbins_to = len(grid) * rate_upsampling
    return np.linspace(grid[0] - 0.5, grid[-1] + 0.5, (nbins_to) * 2 + 1)[1:-1:2]
