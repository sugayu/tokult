'''miscellaneous functions
'''
import numpy as np
from astropy import units as u
import astropy.constants as const
from .common import cosmo


##
def rotate_coord(pos: np.ndarray, angle: float) -> np.ndarray:
    '''Rotate (x,y) coordinates
    Keyword Arguments:
    pos -- position array. shape: (n, m, 2)
    angle -- scalar; angle to rotate. radian
    '''
    rot = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos)


def polar_coord(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    '''Convert (x, y) to polar coordinates (r, phi)
    '''
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi


def fft2(cube: np.ndarray, zero_padding: bool = False) -> np.ndarray:
    '''2 dimensional Fourier transform.
    '''
    shift = (np.array(cube.shape[1:]) / 2.0).astype(int)
    cube_shift = np.roll(cube, shift, axis=(1, 2))
    uvcube = np.fft.fft2(cube_shift)
    uvcube = np.fft.fftshift(uvcube, axes=(1, 2))
    return uvcube


def ifft2(uvcube: np.ndarray) -> np.ndarray:
    '''Inverse 2 dimensional Fourier transform.
    '''
    cube_shift = np.fft.ifftshift(uvcube, axes=(1, 2))
    cube_shift = np.fft.ifft2(cube_shift)
    shift = -(np.array(cube_shift.shape[1:]) / 2.0).astype(int)
    cube = np.roll(cube_shift, shift, axis=(1, 2))
    return cube


def no_lensing(coordinate: np.ndarray) -> np.ndarray:
    '''Dummy function. Return coordinate as itself without lensing.
    '''
    return coordinate


def no_convolve(datacube: np.ndarray, index: int = 0) -> np.ndarray:
    '''Dummy function. Return the datacube as itself without convolution.
    '''
    return datacube


def pixel_scale(pixscale: u.Quantity, redshift: float = 0.0) -> u.Equivalency:
    '''Set pixel scale between pix and arcsec.
    '''
    pixelscale = u.pixel_scale(pixscale)
    Jy_asec2 = u.Jy / (1.0 * u.pix).to(u.arcsec, pixelscale) ** 2
    pixelscale.extend([(u.Jy / u.pix ** 2, u.Unit(Jy_asec2))])

    if redshift > 0.0:
        angdiameter = cosmo.angular_diameter_distance(redshift)
        Mpc_per_pix = (1.0 * u.pix).to(u.rad, pixelscale).value * angdiameter
        pixelscale.extend(
            [
                (u.rad, u.Unit(angdiameter)),
                (u.Jy / u.rad ** 2, u.Unit(u.Jy / angdiameter ** 2)),
                (u.pix, u.Unit(Mpc_per_pix)),
                (u.Jy / u.pix ** 2, u.Unit(u.Jy / Mpc_per_pix ** 2)),
            ]
        )
    return pixelscale


def vpixel_scale(vpixscale: u.Quantity) -> u.Equivalency:
    '''Set velocity-pixel scale between v-pix and km/s.
    '''
    vpixelscale = u.pixel_scale(vpixscale)
    return vpixelscale


def diskmass_scale(
    pixelscale: u.Equivalency, vpixelscale: u.Equivalency
) -> u.Equivalency:
    '''Set disk-mass scale between pix*v-pix**2 and m*km/s**2.
    '''
    m_pix = (1.0 * u.pix).to(u.m, pixelscale)
    kms_vpix = (1.0 * u.pix).to(u.km / u.s, vpixelscale)
    diskmass = (1.0 * m_pix * kms_vpix ** 2 / const.G).decompose()
    return u.Equivalency([(u.Unit(u.pix ** 3), u.Unit(diskmass))])
