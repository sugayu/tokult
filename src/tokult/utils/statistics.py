'''Statistics Utilities
'''

import numpy as np
from numpy.random import default_rng
from typing import Optional, Union


##
def rms(
    cube: np.ndarray, axis: Optional[tuple[int, ...]] = None
) -> Union[np.ndarray, float]:
    '''Compute r.m.s.'''
    sumsq = np.nansum(cube**2, axis=axis)
    cube_zerofill = np.copy(cube)
    cube_zerofill[not np.isfinite(cube)] = 0.0
    n = np.count_nonzero(cube_zerofill, axis=axis)
    return np.sqrt(sumsq / n)


def fft2(cube: np.ndarray) -> np.ndarray:
    '''2 dimensional Fourier transform.'''
    # shift = (np.array(cube.shape[1:]) / 2.0).astype(int)
    # cube_shift = np.roll(cube, shift, axis=(1, 2))
    cube_shift = np.fft.ifftshift(cube, axes=(1, 2))
    uvcube = np.fft.fft2(cube_shift, norm='forward')
    uvcube = np.fft.fftshift(uvcube, axes=(1, 2))
    return uvcube


def ifft2(uvcube: np.ndarray) -> np.ndarray:
    '''Inverse 2 dimensional Fourier transform.'''
    cube_shift = np.fft.ifftshift(uvcube, axes=(1, 2))
    cube_shift = np.fft.ifft2(cube_shift, norm='forward')
    cube = np.fft.fftshift(cube_shift, axes=(1, 2))
    # shift = -(np.array(cube_shift.shape[1:]) / 2.0).astype(int)
    # cube = np.roll(cube_shift, shift, axis=(1, 2))
    return cube


def rfft2(cube: np.ndarray) -> np.ndarray:
    '''2 dimensional real Fourier transform.'''
    cube_shift = np.fft.ifftshift(cube, axes=(1, 2))
    uvcube = np.fft.rfft2(cube_shift, norm='forward')
    uvcube = np.fft.fftshift(uvcube, axes=1)
    return uvcube


def irfft2(uvcube: np.ndarray) -> np.ndarray:
    '''Inverse 2 dimensional real Fourier transform.'''
    cube_shift = np.fft.ifftshift(uvcube, axes=1)
    cube_shift = np.fft.irfft2(cube_shift, norm='forward')
    cube = np.fft.fftshift(cube_shift, axes=(1, 2))
    return cube


def fftconvolve(
    image: np.ndarray,
    kernel: np.ndarray,
    uvcoverage: Optional[np.ndarray] = None,
    axes: Optional[tuple[int, ...]] = None,
) -> np.ndarray:
    '''Wrapper of scipy.signal.fftconvolve

    This function fixes a bug, a difference of central pixels
    between np.fft.fft2 and sp.signal.fftconvolve.
    '''
    size = image[0, :, :].size
    uv = rfft2(image)
    uvpsf = rfft2(kernel)
    uv_noise = size * uv * uvpsf

    if uvcoverage is not None:
        uv_noise[np.logical_not(uvcoverage)] = 0.0

    return irfft2(uv_noise)

    # image_full = sp_fftconvolve(image, kernel, mode='full', axes=axes)

    # # The following code is a modification of signaltools._centered in scipy.

    # # Return the center newshape portion of the array.
    # newshape = np.asarray(image.shape)
    # currshape = np.array(image_full.shape)
    # endind = currshape - (currshape - newshape) // 2
    # startind = endind - newshape
    # myslice = [slice(startind[k], endind[k]) for k in range(len(endind))]
    # return image_full[tuple(myslice)]


def fftconvolve_noise(
    noise: np.ndarray, kernel: np.ndarray, uvcoverage: Optional[np.ndarray] = None
) -> np.ndarray:
    '''Convolve noise like scipy.signal.fftconvolve after modifying kernel.

    Convolution kernel for sky images and noises is different.
    '''
    size = noise[0, :, :].size
    uv = rfft2(noise)
    uvpsf = rfft2(kernel)
    uv_noise = size * uv * np.sqrt(abs(uvpsf.real))

    if uvcoverage is not None:
        uv_noise[np.logical_not(uvcoverage)] = 0.0

    return irfft2(uv_noise)


def create_uvnoise_standardgauss(
    size: tuple[int, ...], seed: Optional[int] = None
) -> np.ndarray:
    '''Create a Gauss noise in the frequency domain.'''
    rng = default_rng(seed)
    noise_real = rng.standard_normal(size=size)
    noise_imag = rng.standard_normal(size=size)
    return noise_real + noise_imag * 1j


def min_abs(array: np.ndarray) -> float:
    '''Return the positive value most close to zero, but not zero.'''
    return abs(array[abs(array) > 0]).min().real
