'''miscellaneous functions
'''
import numpy as np
from numpy.random import default_rng
from astropy import units as u
import astropy.constants as const
from typing import Optional, Union
from .common import cosmo

__all__: list = []


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


def rotate_coord(pos: np.ndarray, angle: float) -> np.ndarray:
    '''Rotate (x,y) coordinates
    Keyword Arguments:
    pos -- position array. shape: (n, m, 2)
    angle -- scalar; angle to rotate. radian
    '''
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos, -1)


def polar_coord(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Convert (x, y) to polar coordinates (r, phi)'''
    r = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y, x)
    return r, phi


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


# def no_lensing(coordinate: np.ndarray) -> np.ndarray:
#     '''Dummy function. Return coordinate as itself without lensing.
#     '''
#     return coordinate


def no_lensing(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, ...]:
    '''Dummy function. Return coordinate as itself without lensing.'''
    return (x, y)


def no_lensing_interpolation(x: float, y: float) -> np.ndarray:
    '''Dummy function. Return coordinate as itself without lensing.'''
    return np.array([x, y])


def no_convolve(datacube: np.ndarray, index: int = 0) -> np.ndarray:
    '''Dummy function. Return the datacube as itself without convolution.'''
    return datacube


def min_abs(array: np.ndarray) -> float:
    '''Return the positive value most close to zero, but not zero.'''
    return abs(array[abs(array) > 0]).min().real


def pixel_scale(pixscale: u.Quantity, redshift: float = 0.0) -> u.Equivalency:
    '''Set pixel scale between pix and arcsec.'''
    pixelscale = u.pixel_scale(pixscale)
    Jy_asec2 = u.Jy / (1.0 * u.pix).to(u.arcsec, pixelscale) ** 2
    pixelscale.extend([(u.Jy / u.pix**2, u.Unit(Jy_asec2))])

    if redshift > 0.0:
        angdiameter = cosmo.angular_diameter_distance(redshift)
        Mpc_per_pix = (1.0 * u.pix).to(u.rad, pixelscale).value * angdiameter
        pixelscale.extend(
            [
                (u.rad, u.Unit(angdiameter)),
                (u.Jy / u.rad**2, u.Unit(u.Jy / angdiameter**2)),
                (u.pix, u.Unit(Mpc_per_pix)),
                (u.Jy / u.pix**2, u.Unit(u.Jy / Mpc_per_pix**2)),
            ]
        )
    return pixelscale


def vpixel_scale(vpixscale: u.Quantity) -> u.Equivalency:
    '''Set velocity-pixel scale between v-pix and km/s.'''
    vpixelscale = u.pixel_scale(vpixscale)
    return vpixelscale


def diskmass_scale(
    pixelscale: u.Equivalency, vpixelscale: u.Equivalency
) -> u.Equivalency:
    '''Set disk-mass scale between pix*v-pix**2 and m*km/s**2.'''
    m_pix = (1.0 * u.pix).to(u.m, pixelscale)
    kms_vpix = (1.0 * u.pix).to(u.km / u.s, vpixelscale)
    diskmass = (1.0 * m_pix * kms_vpix**2 / const.G).decompose()
    return u.Equivalency([(u.Unit(u.pix**3), u.Unit(diskmass))])
