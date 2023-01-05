'''Common utilities
'''
from __future__ import annotations
import os
import pathlib
from dataclasses import dataclass, field
from typing import Optional, Any
import numpy as np
from astropy.cosmology import FlatLambdaCDM
import astropy.constants as astroconst
import astropy.units as u
from astropy.io import fits
from emcee.moves import DEMove, DESnookerMove
import logging
from logging.config import dictConfig
from configparser import ConfigParser

__all__: list = []


##
@dataclass
class DebugParameters:
    mc_dname_savecube = './'
    mc_tag_savecube = 'mc'

    def mc_savecube(self, cube: np.ndarray, i: int):
        '''Save cubes created in Monte Carlo steps.
        '''
        fsave = self.mc_tag_savecube + f'{i:06d}.fits'
        fits.writeto(self.mc_dname_savecube + fsave, cube, overwrite=True)


@dataclass
class ConfigParameters:
    '''Configulation containing hyper parameters.
    '''

    mcmc_init_dispersion: float = 0.001
    mcmc_moves: list = field(
        default_factory=lambda: [(DEMove(), 0.8), (DESnookerMove(), 0.2)]
    )
    noisescale_factor: float = 1.0
    _debug_mode: bool = False
    _debug: DebugParameters = field(default_factory=DebugParameters)


@dataclass(frozen=True)
class Config:
    '''Configulation class containing constants.
    '''

    project: str
    fname_gamma1: Optional[str] = None
    fname_gamma2: Optional[str] = None
    fname_kappa: Optional[str] = None
    fname_ms: Optional[str] = None
    field_id: Optional[str] = None
    nu_obs: Optional[str] = None
    nu_width: Optional[str] = None
    field_id_split: Optional[str] = None
    weight_cont: Optional[str] = None
    cycle: Optional[str] = None
    dv: Optional[str] = None
    z: Optional[float] = None
    pixsize: Optional[str] = None
    num_pix: Optional[int] = None
    vel_start: Optional[str] = None
    weight_cube: Optional[str] = None
    region: Optional[str] = None
    chan_start: Optional[int] = None
    chan_end: Optional[int] = None
    rms_factor: float = 3.0
    specfitsn: float = 5.0
    aparture_rad: int = 1
    mom0SN: int = 3
    ndivide: int = 40
    kernel_num: int = 21
    angular_distance: float = 940.1
    pixkpc: float = angular_distance * np.radians(0.05 / 3600) * 1000
    mpix: float = 1 / 3.085677581e19 / pixkpc
    ngrid: int = 1

    @classmethod
    def from_configparser(cls, file_init: str) -> Config:
        '''Return instance of Config with reading parameters from config.ini'''
        config_ini = cls.configparser(file_init)
        conf_dict = cls.convert_config2dict(config_ini)
        conf_dict_any = cls.convert_configtypes(conf_dict)
        return cls(**conf_dict_any)

    @staticmethod
    def configparser(file_init: str) -> ConfigParser:
        '''wrapper of ConfigParser.'''
        config_ini = ConfigParser(inline_comment_prefixes='#')
        if not os.path.exists(file_init):
            logger.warning('No config.ini file: ' + file_init)
            raise FileNotFoundError
        config_ini.read(file_init, encoding='utf-8')
        return config_ini

    @staticmethod
    def convert_config2dict(config: ConfigParser) -> dict[str, str]:
        '''Convert data types of configure.'''
        list_section = ['glavlens', 'casa']
        output_dict: dict[str, str] = dict(config['DEFAULT'].items())
        for section in list_section:
            if config.has_section(section):
                conf = config[section]
                output_dict.update(**dict(conf.items()))
        return output_dict

    @staticmethod
    def convert_configtypes(config_dict: dict[str, str]) -> dict[str, Any]:
        '''Convert data types of configure.'''
        keytypes = {'z': float, 'num_pix': int, 'chan_start': int, 'chan_end': int}
        for k, f in keytypes.items():
            if k in config_dict.keys():
                config_dict[k] = f(config_dict[k])
        return config_dict


class FileManager:
    '''Manage all the file names used in tokult.'''

    def __init__(self, config: Config) -> None:
        self.conf: Config = config
        self.project: str = self.conf.project
        self.dname_project: str = self.project + '/'
        self.log: str = self.dname_project + 'tokult.log'

    def readconfig_gravlens(self) -> None:
        '''Read config about gravitational lensing to define filename variables'''
        self.dname_gl: str = 'gravlens/'
        self.gamma1: str = 'gamma1.fits'
        self.gamma2: str = 'gamma2.fits'
        self.kappa: str = 'kappa.fits'
        self.gravlens: str = 'gravlens.txt'

    def readconfig_casa(self) -> None:
        '''Read config about CASA analyses to define filename variables'''
        if self.conf.fname_ms is None:
            raise ValueError('self.conf.fname_ms is None.')
        self.ms: str = self.conf.fname_ms
        self.mstxt: str = self.ms + '.txt'
        self.mssplit: str = self.ms + '.split'
        self.mssplittxt: str = self.mssplit + '.txt'
        self.cont_dirty_img: str = 'cont_dirty.image'
        self.cont_clean_img: str = 'cont_clean.image'
        self.cube_dirty_img: str = 'cube_dirty.image'
        self.cube_clean_img: str = 'cube_clean.image'
        self.line_dirty_mom0: str = 'line_dirty_mom0.image'
        self.line_dirty_mom1: str = 'line_dirty_mom1.image'
        self.line_dirty_mom2: str = 'line_dirty_mom2.image'
        self.line_clean_mom0: str = 'line_clean_mom0.image'
        self.line_clean_mom1: str = 'line_clean_mom1.image'
        self.line_clean_mom2: str = 'line_clean_mom2.image'
        self.line_dirty_mom0_fits: str = 'line_dirty_mom0.fits'
        self.line_dirty_mom1_fits: str = 'line_dirty_mom1.fits'
        self.line_dirty_mom2_fits: str = 'line_dirty_mom2.fits'
        self.line_clean_mom0_fits: str = 'line_clean_mom0.fits'
        self.line_clean_mom1_fits: str = 'line_clean_mom1.fits'
        self.line_clean_mom2_fits: str = 'line_clean_mom2.fits'
        self.init_para: str = 'init_param.dat'
        self.rms: str = 'rms.dat'
        self.chi2: str = 'chi2.dat'
        self.velocity: str = 'velocity.dat'
        self.sigma: str = 'sigma.dat'
        self.mom0_highSN: str = 'mom0_highSN.dat'
        self.mom0_dev: str = 'mom0_dev.dat'
        self.mom0_ch: str = 'mom0_ch.dat'
        self.output: str = 'cubepixels.dat'
        self.cube_dev: str = 'cube_dev.dat'
        self.beam: str = 'beam.image'
        self.beam_fits: str = 'beam.fits'
        self.beam_dat: str = 'beam.dat'
        self.bestfit: str = 'bestfit.dat'
        self.dname_cont: str = self.dname_project + 'cont/'
        self.dname_cube: str = self.dname_project + 'cube/'
        self.dname_spec: str = self.dname_cube + 'spectra/'
        self.dname_mom: str = self.dname_project + 'mom/'
        self.dname_init_para: str = self.dname_project + 'init_para/'
        self.dname_result: str = self.dname_project + 'result/'


@dataclass(frozen=True)
class Const:
    # pixkpc = angular_distance * np.radians(pixsize / 3600) * 1000
    kpcm: float = 3.085677581e19
    # mpix = 1 / kpcm / pixkpc
    kgMo: float = 1 / (1.989e30)
    G: u.Quantity = astroconst.G


def initialize_tokult(file_init: Optional[str] = None) -> None:
    '''Initialize tokult.
    Specifically, set global parameters: conf, fname, logger
    '''
    global conf, fnames, logger
    if file_init is None:
        file_init = os.path.dirname(__file__) + '/config.ini'
    conf = Config.from_configparser(file_init)
    fnames = FileManager(conf)

    path = pathlib.Path(fnames.dname_project)
    path.mkdir(exist_ok=True)

    dictconfig_logging = get_dictconfig_logging(fnames.log)
    dictConfig(dictconfig_logging)
    logger = logging.getLogger(__name__)


def get_dictconfig_logging(fnames_log: str) -> dict[str, Any]:
    '''Get dictconfig for logger'''
    return {
        'version': 1,
        'formatters': {
            'myfmt': {
                'format': '[%(asctime)s] %(filename)s %(funcName)s: %(levelname)s - %(message)s'
            }
        },
        'handlers': {
            'consoleERRORHand': {
                'class': 'logging.StreamHandler',
                'level': 'ERROR',
                'formatter': 'myfmt',
            },
            'fileHandler': {
                'class': 'logging.FileHandler',
                'level': 'INFO',
                'formatter': 'myfmt',
                'filename': fnames_log,
            },
            'consoleHandler': {'class': 'logging.StreamHandler', 'formatter': 'myfmt'},
            'fileDebugHandler': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'myfmt',
                'filename': fnames_log,
            },
        },
        'root': {'level': 'INFO', 'handlers': ['fileHandler', 'consoleERRORHand']},
        'loggers': {
            'debuglog': {
                'level': 'DEBUG',
                'handlers': ['fileDebugHandler', 'consoleHandler'],
                'propagate': False,
            }
        },
        'disable_existing_loggers': False,
    }


# global variables
const: Const = Const()
conf = Config('temp')
fnames = FileManager(conf)
logger = logging.getLogger(__name__)
cosmo = FlatLambdaCDM(H0=70, Om0=0.3, Ob0=0.04)
