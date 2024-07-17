'''Tokult --- Tools of Kinematics Used for Lensed Targets

Tokult is a kinematics fitting tool.
'''

from .__version import __version__
from .ui import Tokult
from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.04)

__all__ = ['__version__', 'Tokult']
