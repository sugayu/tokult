'''Modules of functions
'''

import numpy as np
import scipy.special as sps


##
def gaussian(x, center, sigma, area):
    '''Gaussian function.
    '''
    norm = area / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def reciprocal_exp(r, norm, rnorm):
    '''Exponential function
    '''
    return norm * np.exp(-r / rnorm)


def freeman_disk(r, phi, mass_dyn, rnorm, incl):
    '''Freeman disk function
    '''
    r2h = 0.5 * r / rnorm
    myu_0 = mass_dyn / (2 * np.pi * rnorm ** 2)
    G = 1
    A = sps.iv(0, r2h) * sps.kv(0, r2h) - sps.iv(1, r2h) * sps.kv(1, r2h)
    f_sightline = np.cos(phi) * np.sin(incl)
    velocity = np.sqrt(4 * np.pi * G * myu_0 * rnorm * r2h ** 2 * A) * f_sightline
    return velocity
