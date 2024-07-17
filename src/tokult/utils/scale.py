'''Funcitons to convert scales.
'''

from astropy import units as u
import astropy.constants as const
from .. import cosmo


##
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
