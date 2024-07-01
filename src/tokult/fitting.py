'''Modules of fitting functions
'''
from __future__ import annotations
from dataclasses import dataclass, field

# import time
# import pickle
# from pathlib import Path
import numpy as np
from numpy.random import default_rng
from scipy.optimize import least_squares as sp_least_squares
from scipy.optimize.optimize import OptimizeResult
from astropy.io import fits
from astropy import units as u
from astropy.wcs import WCS
import tqdm
from typing import Callable, Sequence, Optional, Union, TYPE_CHECKING
from typing import NamedTuple
import emcee
from multiprocessing.pool import Pool
from . import function as func
from . import misc
from . import common as c
