'''Core modules of tokult.
'''
from __future__ import annotations
import numpy as np
from scipy.signal import convolve2d
from astropy.io import fits
from astropy import wcs
from typing import Callable, Sequence, Optional, Union
from numpy.typing import ArrayLike
from . import common as c
from . import fitting
from .misc import fft2


##
class Tokult:
    '''Main class in tokult package.

    Users manage tokult methods via this class.
    '''

    def __init__(
        self,
        datacube: DataCube,
        dirtybeam: Optional[DirtyBeam] = None,
        gravlens: Optional[GravLens] = None,
    ) -> None:
        self.datacube = datacube
        self.dirtybeam = dirtybeam
        self.gravlens = gravlens
        self.modelcube: Optional[ModelCube] = None

    @classmethod
    def launch(
        cls,
        data: Union[np.ndarray, str],
        beam: Union[np.ndarray, str, None] = None,
        gravlens: Union[
            tuple[np.ndarray, np.ndarray, np.ndarray], tuple[str, str, str], None
        ] = None,
        header_data: Optional[fits.Header] = None,
        header_beam: Optional[fits.Header] = None,
        header_gamma: Optional[fits.Header] = None,
        index_data: int = 0,
        index_beam: int = 0,
        index_gamma: int = 0,
    ):
        '''Constructer of Tokult.
        '''
        datacube = DataCube.create(data, header=header_data, index_hdul=index_data)

        dirtybeam: Optional[DirtyBeam]
        if beam:
            dirtybeam = DirtyBeam.create(
                beam, header=header_beam, index_hdul=index_beam
            )
        else:
            dirtybeam = None

        gl: Optional[GravLens]
        if gravlens is not None:
            g1, g2, k = gravlens
            gl = GravLens.create(g1, g2, k, header=header_gamma, index_hdul=index_gamma)
        else:
            gl = None
        return cls(datacube, dirtybeam, gl)

    def imagefit(
        self,
        init: Sequence[float],
        bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
        niter: int = 1,
        fix: Optional[fitting.FixParams] = None,
        is_separate: bool = False,
    ) -> fitting.Solution:
        '''First main function to fit 3d model to data cube on image plane.
        '''
        func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
        func_lensing = self.gravlens.lensing if self.gravlens else None

        solution = fitting.least_square(
            self.datacube,
            init,
            bound,
            func_convolve=func_convolve,
            func_lensing=func_lensing,
            niter=niter,
            mode_fit='image',
            fix=fix,
            is_separate=is_separate,
        )
        self.construct_modelcube(solution.best)
        return solution

    def uvfit(
        self,
        init: Sequence[float],
        bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
        fix: Optional[fitting.FixParams] = None,
        is_separate: bool = False,
    ) -> fitting.Solution:
        '''Second main function to fit 3d model to data cube on uv plane.
        '''
        if self.dirtybeam:
            beam_visibility = self.dirtybeam.uvplane
        else:
            msg = '"DirtyBeam" is necessarily for uvfit.'
            c.logger.warning(msg)
            raise ValueError(msg)
        func_lensing = self.gravlens.lensing if self.gravlens else None

        solution = fitting.least_square(
            self.datacube,
            init,
            bound,
            beam_vis=beam_visibility,
            func_lensing=func_lensing,
            mode_fit='uv',
            fix=fix,
            is_separate=is_separate,
        )
        self.construct_modelcube(solution.best)
        return solution

    def initialguess(self) -> fitting.InputParams:
        '''Guess initial input parameters for fitting.
        '''
        func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
        func_lensing = self.gravlens.lensing if self.gravlens else None
        return fitting.initialguess(self.datacube, func_convolve, func_lensing)

    def set_region(
        self,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        '''Set region of datacube used for fitting.
        '''
        self.datacube.cutout(xlim, ylim, vlim)
        if self.dirtybeam is not None:
            self.dirtybeam.cutout_to_match_with(self.datacube)
        if self.gravlens is not None:
            self.gravlens.match_wcs_with(self.datacube)

    def set_datacube(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        '''Set datacube into DataCube class.
        '''
        self.datacube = DataCube.create(
            data_or_fname, header=header, index_hdul=index_hdul
        )
        self.set_region(xlim, ylim, vlim)

    def set_dirtybeam(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        '''Set dirtybeam data into DirtyBeam class.
        '''
        self.dirtybeam = DirtyBeam.create(
            data_or_fname, header=header, index_hdul=index_hdul
        )

    def set_gravlens(
        self,
        data_or_fname: Union[
            tuple[np.ndarray, np.ndarray, np.ndarray], tuple[str, str, str]
        ],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        '''Set gravitational lensing data into GravLens class.
        '''
        g1, g2, k = data_or_fname
        self.gravlens = GravLens.create(g1, g2, k, header=header, index_hdul=index_hdul)

    def construct_modelcube(self, params: tuple[float, ...]) -> None:
        '''Construct model cube using parameters.
        '''
        datacube = self.datacube
        func_lensing = self.gravlens.lensing if self.gravlens else None
        func_convolve = self.dirtybeam.fullconvolve if self.dirtybeam else None
        # fitting.initialize_globalparameters_for_image(
        #     datacube, func_convolve, func_lensing
        # )
        self.modelcube = ModelCube.create(
            params, datacube=datacube, convolve=func_convolve, lensing=func_lensing
        )


class Cube:
    '''Contains cube data and related methods.
    '''

    def __init__(
        self,
        imageplane: np.ndarray,
        header: Optional[fits.Header] = None,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        self.original = imageplane
        self.imageplane = imageplane
        self.header = header
        self.uvplane = self.fft2(self.original, zero_padding=True)

        self.xlim: tuple[int, int]
        self.ylim: tuple[int, int]
        self.vlim: tuple[int, int]
        self.vgrid: np.ndarray
        self.xgrid: np.ndarray
        self.ygrid: np.ndarray
        self.coord_imageplane: list[np.ndarray]
        self.cutout(xlim, ylim, vlim)

    def cutout(
        self,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        '''Create coordinates of data cube.
        '''
        self.xlim = xlim if xlim else (0, self.original.shape[2])
        self.ylim = ylim if ylim else (0, self.original.shape[1])
        self.vlim = vlim if vlim else (0, self.original.shape[0])
        self.xslice = slice(*self.xlim)
        self.yslice = slice(*self.ylim)
        self.vslice = slice(*self.vlim)

        xarray = np.arange(self.xlim[0], self.xlim[1])
        yarray = np.arange(self.ylim[0], self.ylim[1])
        varray = np.arange(self.vlim[0], self.vlim[1])
        self.coord_imageplane = np.meshgrid(varray, yarray, xarray, indexing='ij')
        self.vgrid, self.ygrid, self.xgrid = self.coord_imageplane
        self.imageplane = self.original[self.vslice, self.yslice, self.xslice]
        self.uvplane = self.fft2(self.original[self.vslice, :, :], zero_padding=True)

    def rms(self):
        '''Return rms noise of the datacube.
        '''
        maskedimage = np.copy(self.original[self.vslice, :, :])
        maskedimage[:, self.yslice, self.xslice] = None
        sumsq = np.nansum(maskedimage ** 2, axis=(1, 2))
        n = np.count_nonzero(maskedimage, axis=(1, 2))
        return np.sqrt(sumsq / n)

    def moment0(self) -> np.ndarray:
        '''Return moment 0 maps using pixel indicies.
        '''
        return np.sum(self.imageplane, axis=0)

    def rms_moment0(self):
        '''Return rms noise of the moment 0 map.
        '''
        maskedimage = self.moment0()
        maskedimage[self.yslice, self.xslice] = None
        sumsq = np.nansum(maskedimage ** 2)
        n = np.count_nonzero(maskedimage)
        return np.sqrt(sumsq / n)

    def pixmoment1(self, thresh: float = 0.0) -> np.ndarray:
        '''Return moment 1 maps using pixel indicies.

        This funciton uses pixel indicies instead of velocity; that is,
        the units of moment 1 is pixel.
        '''
        mom0 = self.moment0()
        mom1 = np.sum(self.imageplane * self.vgrid, axis=0) / mom0
        mom1[mom0 <= thresh] = None
        return mom1

    def pixmoment2(self, thresh: float = 0.0) -> np.ndarray:
        '''Return moment 1 maps using pixel indicies.

        This funciton uses pixel indicies instead of velocity; that is,
        the units of moment 2 maps is pixel.
        '''
        mom0 = self.moment0()
        mom1 = self.pixmoment1()
        vv = self.vgrid - mom1[np.newaxis, ...]
        mom2 = np.sum(self.imageplane * np.sqrt(vv ** 2), axis=0) / mom0
        mom2[mom0 <= thresh] = None
        return mom2

    def get_pixmoments(
        self, imom: int = 0, thresh: float = 0.0, recalc: bool = False
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
            self.mom1 = np.sum(self.imageplane * self.vgrid, axis=0) / mom0
            self.mom1[mom0 <= thresh] = None
            return self.mom1
        if imom == 2:
            if not recalc:
                try:
                    return self.mom2
                except AttributeError:
                    pass
            mom1 = self.get_pixmoments(imom=1)
            vv = self.vgrid - mom1[np.newaxis, ...]
            self.mom2 = np.sum(self.imageplane * np.sqrt(vv ** 2), axis=0) / self.mom0
            self.mom2[self.mom0 <= thresh] = None
            return self.mom2

        message = 'An input "imom" should be the int type of 0, 1, or 2.'
        c.logger.error(message)
        raise ValueError(message)

    @staticmethod
    def fft2(data: np.ndarray, zero_padding: bool = False) -> np.ndarray:
        '''Wrapper of misc.fft2.
        '''
        if np.any(idx := (np.logical_not(np.isfinite(data)))):
            if zero_padding:
                data[idx] = 0.0
            else:
                raise ValueError(
                    'Input cube data includes non-finite values (NaN or Inf).'
                )
        return fft2(data)


class DataCube(Cube):
    '''Cube class to contain an observed datacube.
    '''

    @classmethod
    def create(
        cls,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> DataCube:
        '''Constructer.
        '''
        if isinstance(data_or_fname, np.ndarray):
            return cls(data_or_fname, header)
        elif isinstance(data_or_fname, str):
            imageplane, header = cls.loadfits(data_or_fname, index_hdul=index_hdul)
            return cls(imageplane, header)
        message = (
            f'The first input must be np.ndarray or str, '
            f'but the input type is {type(data_or_fname)}.'
        )
        c.logger.error(message)
        raise TypeError(message)

    @staticmethod
    def loadfits(fname: str, index_hdul: int = 0) -> tuple[np.ndarray, fits.Header]:
        '''Read cube data from fits file.
        '''
        with fits.open(fname) as hdul:
            # np.squeeze is needed to erase the polari axis.
            imageplane = np.squeeze(hdul[index_hdul].data)
            header = hdul[index_hdul].header
        return imageplane, header


class ModelCube(Cube):
    '''Cube class to contain a modeled datacube.
    '''

    @classmethod
    def create(
        cls,
        params: tuple[float, ...],
        datacube: DataCube,
        lensing: Optional[Callable] = None,
        convolve: Optional[Callable] = None,
    ) -> ModelCube:
        '''Constructer.
        '''
        # shape = datacube.original.shape
        # x, y, v = (np.arange(shape[2]), np.arange(shape[1]), np.arange(shape[0]))
        # vv_grid, yy_grid, xx_grid = np.meshgrid(v, y, x)
        imagecube = fitting.construct_model_at_imageplane_with(
            params,
            xx_grid=datacube.xgrid,
            yy_grid=datacube.ygrid,
            vv_grid=datacube.vgrid,
            lensing=lensing,
        )
        modelcube = np.zeros_like(datacube.original)
        xs, ys, vs = datacube.xslice, datacube.yslice, datacube.vslice
        modelcube[vs, ys, xs] = imagecube

        if convolve is not None:
            model_convolved = np.empty_like(modelcube)
            for i, image in enumerate(modelcube):
                model_convolved[i, :, :] = convolve(image, index=i)

        xlim, ylim, vlim = datacube.xlim, datacube.ylim, datacube.vlim
        return cls(model_convolved, xlim=xlim, ylim=ylim, vlim=vlim)


class DirtyBeam:
    '''Contains dirtybeam, i.e., Point Spread Functions (PSF) of Cube.

    Contained data is dirtybeam images as a function of frequency.
    '''

    def __init__(self, beam: np.ndarray, header: Optional[fits.Header] = None,) -> None:
        self.original = beam
        self.imageplane = beam
        self.header = header
        self.uvplane = fft2(self.original)

    @classmethod
    def create(
        cls,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> DirtyBeam:
        '''Constructer
        '''
        if isinstance(data_or_fname, np.ndarray):
            return cls(data_or_fname, header)
        elif isinstance(data_or_fname, str):
            beam, header = cls.loadfits(data_or_fname, index_hdul=index_hdul)
            return cls(beam, header)
        message = (
            f'The first input must be np.ndarray or str, '
            f'but the input type is {type(data_or_fname)}.'
        )
        c.logger.error(message)
        raise TypeError(message)

    def convolve(self, image: np.ndarray, index=0) -> np.ndarray:
        '''Convolve image with dirtybeam (psf).
        '''
        # s1 = np.arange(c.conf.kernel_num)
        # t1 = np.arange(c.conf.kernel_num)
        # s2, t2 = np.meshgrid(s1, t1)
        # s3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + s2
        # t3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + t2
        # st = np.array(c.conf.num_pix * s3 + t3, dtype=int)
        # kernel = self.beam[st]
        # kernel2 = kernel / np.sum(kernel)
        kernel = self.imageplane[index, :, :]
        # kernel = beam  / np.sum(beam)

        # todo: should be changed to fftconvolve
        return convolve2d(image, kernel, mode='same')

    def fullconvolve(self, image: np.ndarray, index=0) -> np.ndarray:
        '''Convolve image with original-size dirtybeam (psf).
        '''
        kernel = self.original[index, :, :]
        # todo: should be changed to fftconvolve
        return convolve2d(image, kernel, mode='same')

    def cutout(
        self,
        xlim: Union[tuple[int, int], slice],
        ylim: Union[tuple[int, int], slice],
        vlim: Union[tuple[int, int], slice],
    ) -> None:
        '''Cutout a cubic region from the dirty beam map.
        '''
        xslice = slice(*xlim) if isinstance(xlim, tuple) else xlim
        yslice = slice(*ylim) if isinstance(ylim, tuple) else ylim
        vslice = slice(*vlim) if isinstance(vlim, tuple) else vlim
        self.imageplane = self.original[vslice, yslice, xslice]
        self.uvplane = fft2(self.original[vslice, :, :])

    def cutout_to_match_with(self, cube: DataCube) -> None:
        '''Cutout a cubic region from the dirty beam map.
        '''
        xslice = cube.xslice
        yslice = cube.yslice
        vslice = cube.vslice
        self.cutout(xslice, yslice, vslice)

    @staticmethod
    def loadfits(fname: str, index_hdul: int = 0) -> tuple[np.ndarray, fits.Header]:
        '''Read dirtybeam from fits file.

        CAUTION:
        The data shape of dirtybeam must be the same as the datacube (cubeimage).
        This requirement is naturaly satisfied if input dirtybeam and cubeimage fits
        files are simulationsly created.
        '''
        with fits.open(fname) as hdul:
            beam = hdul[index_hdul].data
            header = hdul[index_hdul].header
        return np.squeeze(beam), header


class GravLens:
    '''Contains gravitational lensing used for Cube.

    Contents are lensing parameters depending on positions: gamma1, gamma2, and kappa.
    '''

    def __init__(
        self,
        gamma1: np.ndarray,
        gamma2: np.ndarray,
        kappa: np.ndarray,
        header: Optional[fits.Header] = None,
    ) -> None:
        self.original_gamma1 = gamma1
        self.original_gamma2 = gamma2
        self.original_kappa = kappa
        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kappa = kappa
        self.header = header

    @classmethod
    def create(
        cls,
        data_or_fname_gamma1: Union[np.ndarray, str],
        data_or_fname_gamma2: Union[np.ndarray, str],
        data_or_fname_kappa: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> GravLens:
        '''Constructer
        '''
        if not (
            isinstance(data_or_fname_gamma1, type(data_or_fname_gamma2))
            and isinstance(data_or_fname_gamma1, type(data_or_fname_kappa))
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
            gamma1 = data_or_fname_gamma1
            gamma2 = data_or_fname_gamma2
            kappa = data_or_fname_kappa
            return cls(gamma1, gamma2, kappa, header)
        elif isinstance(data_or_fname_gamma1, str):
            assert isinstance(data_or_fname_gamma2, str)
            assert isinstance(data_or_fname_kappa, str)
            loaded = cls.loadfits(
                data_or_fname_gamma1,
                data_or_fname_gamma2,
                data_or_fname_kappa,
                index_hdul=index_hdul,
            )
            return cls(*loaded)

        message = (
            f'The first input must be np.ndarray or str, '
            f'but the input type is {type(data_or_fname_gamma1)}.'
        )
        c.logger.error(message)
        raise TypeError(message)

    def lensing(self, coordinates: np.ndarray) -> np.ndarray:
        '''Convert coordinates (x, y) from the image plane to the source plane.

        This method use gravitational lensing parameters: gamma1, gamma2, and kappa.
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

    def match_wcs_with(self, cube: DataCube):
        '''Match the world coordinate system with input data.
        '''
        wcs_cube = wcs.WCS(cube.header)
        wcs_gl = wcs.WCS(self.header)
        skycoord_wcs, _, _ = wcs_cube.pixel_to_world(
            cube.xgrid[0, :, :].ravel(), cube.ygrid[0, :, :].ravel(), 0, 0
        )
        idx = wcs_gl.world_to_array_index(skycoord_wcs)
        shape = cube.xgrid.shape[1:]
        self.gamma1 = self.original_gamma1[idx].reshape(*shape)
        self.gamma2 = self.original_gamma2[idx].reshape(*shape)
        self.kappa = self.original_kappa[idx].reshape(*shape)

    @staticmethod
    def loadfits(
        fname_gamma1: str, fname_gamma2: str, fname_kappa: str, index_hdul: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
        '''Read gravlens from fits file.
        '''
        with fits.open(fname_gamma1) as hdul:
            gamma1 = hdul[index_hdul].data
            header = hdul[index_hdul].header
        with fits.open(fname_gamma2) as hdul:
            gamma2 = hdul[index_hdul].data
        with fits.open(fname_kappa) as hdul:
            kappa = hdul[index_hdul].data
        return gamma1, gamma2, kappa, header
