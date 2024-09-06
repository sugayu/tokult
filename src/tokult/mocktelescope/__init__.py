'''Mock observation of the galaxy model to return an observed data cube.
'''

from .telescope import MockTelescope
from .psfconvolve import PointSpreadFunction

__all__ = ['MockTelescope', 'PointSpreadFunction']
