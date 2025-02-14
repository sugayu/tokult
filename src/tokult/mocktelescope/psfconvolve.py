'''PFS convolution.
'''

import numpy as np
from .telescope import TelescopeLayer
from ..utils import statistics
from logging import getLogger

logger = getLogger(__name__)


##
class PointSpreadFunction(TelescopeLayer):
    '''Convolve data cubes with Point Spread Function.'''

    def __init__(self, psf: np.ndarray) -> None:
        if psf.ndim == 2:
            psf = psf[np.newaxis, :, :]
        self.psf = psf
        self.shape = psf.shape

        if not np.isclose(np.sum(psf), 1.0):
            logger.warning(
                f'The sum of the PSF image is not unity, but {np.sum(psf)}. '
                'Is this working as you expected?'
            )

        peak_position = np.unravel_index(np.argmax(psf), self.shape)
        supposed_peak_position = tuple([s // 2 for s in self.shape])
        if not peak_position == supposed_peak_position:
            logger.warning(
                f'The peak position of PSF, {peak_position}, is offset '
                f'from the supposed peak position, {supposed_peak_position}. '
                'Is this working as you expected?'
            )
        super().__init__()

    def __call__(self, sky: np.ndarray) -> np.ndarray:
        return self.convolve(sky)

    def convolve(self, sky: np.ndarray) -> np.ndarray:
        '''Convolve a data cube with the PSF.'''
        if sky.shape[-2:] != self.shape[-2:]:
            msg = (
                f'The shape of the data cube {sky.shape} must be '
                f'the same as the shape of the psf image {self.shape}.'
            )
            logger.error(msg)
            raise ValueError(msg)

        if sky.ndim == 2:
            return statistics.fftconvolve(sky[np.newaxis, :, :], self.psf).squeeze()
        elif sky.ndim == 3:
            return statistics.fftconvolve(sky, self.psf)
        else:
            msg = f'Dimension of image is two or three, not {sky.ndim}.'
            logger.error(msg)
            raise ValueError(msg)
