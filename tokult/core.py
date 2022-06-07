'''Core modules of tokult.
'''
import numpy as np
from scipy.signal import convolve2d
from astropy.io import fits
from typing import Sequence, Optional, Union, Callable
from . import common as c
from . import fitting as fit


##
class Tokult:
    '''Main class in tokult package.

    Users manage tokult methods via this class.
    '''

    def __init__(self) -> None:
        pass

    def image_fit(
        self, init: Sequence[float], bound: Sequence[float], func_fit: Callable
    ):
        '''First main function to fit 3d model to data cube on image plane.
        '''
        sol = fit.least_square(self.datacube, init, bound, func_fit)
        return sol

    def uv_fit(self):
        '''Second main function to fit 3d model to data cube on uv plane.
        '''
        pass

    def set_datacube(
        self,
        fname: str,
        xlim: Optional[tuple[int, ...]] = None,
        ylim: Optional[tuple[int, ...]] = None,
        vlim: Optional[tuple[int, ...]] = None,
    ) -> None:
        '''Set datacube into DataCube class.
        '''
        self.datacube = DataCube(fname)
        self.datacube.coordinate()

    def set_dirtybeam(self, fname: str) -> None:
        '''Set dirtybeam data into DirtyBeam class.
        '''
        self.dirtybeam = DirtyBeam(fname)

    def set_gravlenz(self, gamma1: str, gamma2: str, kappa: str) -> None:
        '''Set gravitational lenzing data into GravLenz class.
        '''
        self.gravlenz = GravLenz(gamma1, gamma2, kappa)


class Cube:
    '''Contains cube data and related methods.
    '''

    def __init__(self) -> None:
        self.imageplane: np.ndarray

    def coordinate(
        self,
        xlim: Optional[tuple[int, ...]] = None,
        ylim: Optional[tuple[int, ...]] = None,
        vlim: Optional[tuple[int, ...]] = None,
    ) -> None:
        '''Create coordinates of data cube.
        '''
        xlow, xup = xlim if xlim else 0, self.imageplane.shape[2] - 1
        ylow, yup = ylim if ylim else 0, self.imageplane.shape[1] - 1
        vlow, vup = vlim if vlim else 0, self.imageplane.shape[0] - 1

        xarray = np.arange(xlow, xup + 1)
        yarray = np.arange(ylow, yup + 1)
        varray = np.arange(vlow, vup + 1)
        self.coord_imageplane = np.meshgrid(varray, xarray, yarray, indexing='ij')

    def get_rms(self):
        '''Return rms noise of the datacube.
        '''
        try:
            return self.rms

        except AttributeError:
            # FIXME: Write code to compute rms as a function of frequency.
            return self.rms

    def get_pixmoments(
        self, imom: int = 0, mask_minmom0: float = 0.0, recalc: bool = False
    ) -> np.ndarray:
        '''Return moment maps using pixel indicies.

        This funciton uses pixel indicies instead of velocity; that is,
        the units of moment 1 and 2 maps are pixel and the moment 0 map is the same as
        the one returned by a method "get_moments".
        '''
        self.mom0: np.ndarray
        self.mom1: np.ndarray
        self.mom2: np.ndarray

        if imom == 0:
            if not recalc:
                try:
                    return self.mom0
                except AttributeError:
                    pass
            self.mom0 = np.sum(self.imageplane, axis=0)
            return self.mom0
        if imom == 1:
            if not recalc:
                try:
                    return self.mom1
                except AttributeError:
                    pass
            mom0 = self.get_pixmoments(imom=0)
            varray = self.coord_imageplane[0]
            self.mom1 = np.sum(self.imageplane * varray, axis=0) / mom0
            self.mom1[mom0 <= mask_minmom0] = None
            return self.mom1
        if imom == 2:
            if not recalc:
                try:
                    return self.mom2
                except AttributeError:
                    pass
            mom1 = self.get_pixmoments(imom=1)
            varray = self.coord_imageplane[0]
            vv = varray - mom1[np.newaxis, ...]
            self.mom2 = np.sum(self.imageplane * np.sqrt(vv ** 2), axis=0) / self.mom0
            self.mom2[self.mom0 <= mask_minmom0] = None
            return self.mom2

        message = 'An input "imom" should be the int type of 0, 1, or 2.'
        c.logger.error(message)
        raise ValueError(message)


class DataCube(Cube):
    '''Cube class to contain an observed datacube.
    '''

    def __init__(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        if isinstance(data_or_fname, np.ndarray):
            self.imageplane = data_or_fname
            self.header = header
        elif isinstance(data_or_fname, str):
            self.readfile(data_or_fname, index_hdul=index_hdul)
        self.coordinate()

    def readfile(self, fname: str, index_hdul: int = 0) -> None:
        '''Read cube data from fits file.
        '''
        with fits.open(fname) as hdul:
            # np.squeeze is needed to erase polari info.
            self.imageplane = np.squeeze(hdul[index_hdul].data)
            self.header = hdul[index_hdul].header


class ModelCube(Cube):
    '''Cube class to contain a modeled datacube.
    '''

    pass


class DirtyBeam:
    '''Contains dirtybeam, i.e., Point Spread Functions (PSF) of Cube.

    Contained data is dirtybeam images as a function of frequency.
    '''

    def __init__(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        if isinstance(data_or_fname, np.ndarray):
            self.pfs = data_or_fname
            self.header = header
        elif isinstance(data_or_fname, str):
            self.readfile(data_or_fname, index_hdul=index_hdul)

    def readfile(self, fname: str, index_hdul: int = 0) -> None:
        '''Read dirtybeam from fits file.

        CAUTION:
        The data shape of dirtybeam must be the same as the datacube (cubeimage).
        This requirement is naturaly satisfied if input dirtybeam and cubeimage fits
        files are simulationsly created.
        '''
        with fits.open(fname) as hdul:
            self.pfs = hdul[index_hdul].data
            self.header = hdul[index_hdul].header

    def convolve(self, image):
        '''Convolve image with dirtybeam (psf).
        '''
        s1 = np.arange(c.conf.kernel_num)
        t1 = np.arange(c.conf.kernel_num)
        s2, t2 = np.meshgrid(s1, t1)
        s3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + s2
        t3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + t2
        st = np.array(c.conf.num_pix * s3 + t3, dtype=int)
        kernel = self.pfs[st]
        kernel2 = kernel / sum(sum(kernel))
        return convolve2d(image, kernel2, 'same')


class GravLenz:
    '''Contains gravitational lenzing used for Cube.

    Contents are lenzing parameters depending on positions: gamma1, gamma2, and kappa.
    '''

    def __init__(
        self,
        data_or_fname_gamma1: Union[np.ndarray, str],
        data_or_fname_gamma2: Union[np.ndarray, str],
        data_or_fname_kappa: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        if not (
            isinstance(data_or_fname_gamma1, type(data_or_fname_gamma2))
            & isinstance(data_or_fname_gamma1, type(data_or_fname_kappa))
        ):
            message = (
                f'Types of inputs for gamma1, gamma2, kappa are different: '
                f'gamma1={type(data_or_fname_gamma1)} '
                f'gamma2={type(data_or_fname_gamma2)} '
                f'kappa={type(data_or_fname_kappa)}'
            )
            c.logger.error(message)
            raise TypeError(message)

        if isinstance(data_or_fname_gamma1, np.ndarray):
            assert isinstance(data_or_fname_gamma2, np.ndarray)
            assert isinstance(data_or_fname_kappa, np.ndarray)
            self.gamma1 = data_or_fname_gamma1
            self.gamma2 = data_or_fname_gamma2
            self.kappa = data_or_fname_kappa
            self.header = header
        elif isinstance(data_or_fname_gamma1, str):
            assert isinstance(data_or_fname_gamma2, str)
            assert isinstance(data_or_fname_kappa, str)
            self.readfile(
                data_or_fname_gamma1,
                data_or_fname_gamma2,
                data_or_fname_kappa,
                index_hdul=index_hdul,
            )

    def readfile(
        self,
        fname_gamma1: str,
        fname_gamma2: str,
        fname_kappa: str,
        index_hdul: int = 0,
    ) -> None:
        '''Read gravlenz from fits file.
        '''
        with fits.open(fname_gamma1) as hdul:
            self.gamma1 = hdul[index_hdul].data
            self.header = hdul[index_hdul].header
        with fits.open(fname_gamma2) as hdul:
            self.gamma2 = hdul[index_hdul].data
        with fits.open(fname_kappa) as hdul:
            self.kappa = hdul[index_hdul].data

    def lenz_image2source(self, coordinates: np.ndarray) -> np.ndarray:
        '''Convert coordinates (x, y) from the image plane to the source plane.

        This method use gravitational lenzing parameters: gamma1, gamma2, and kappa.
        coordinates -- position array including (x, y).
                       shape: (n, m, 2) and shape of x: (n, m)
        '''
        g1 = self.gamma1
        g2 = self.gamma2
        k = self.kappa

        jacob = np.array([[1 - k - g1, -g2], [-g2, 1 - k + g1]])
        axis = np.concatenate((2 + np.arange(g1.ndim), (0, 1)))
        # assert axis == np.array([2, 3, 0, 1])
        jacob = jacob.transpose(axis)
        # assert jacob.shape == (n, m, 2, 2)
        _coord = coordinates[..., np.newaxis]
        return np.squeeze(jacob @ _coord)
