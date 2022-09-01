'''Core modules of tokult.
'''
from __future__ import annotations
import numpy as np
from numpy.random import default_rng
from astropy.io import fits
from astropy import wcs
from multiprocessing.pool import Pool
from typing import Callable, Sequence, Optional, Union
from . import common as c
from . import fitting
from . import misc

__all__ = ['Tokult', 'Cube', 'DataCube', 'ModelCube', 'DirtyBeam', 'GravLens']


##
class Tokult:
    '''Interface of Tokult.

    Users specify data to launch an instance object and start fitting through
    the instance. This class contains observed data, psf, and lensing parameters,
    along with useful functions to manipulate data and models.

    Args:
        datacube (DataCube):
        dirtybeam (Optional[DirtyBeam], optional): Defaults to None.
        gravlens (Optional[GravLens], optional): Defaults to None.

    Examples:
        >>> import tokult
        >>> tok = tokult.Tokult.launch('data.fits', 'psf.fits',
                                       ('gamma1.fits', 'gamma2.fits', 'kappa.fits'))
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
        gravlens: Union[tuple[np.ndarray, ...], tuple[str, ...], None] = None,
        header_data: Optional[fits.Header] = None,
        header_beam: Optional[fits.Header] = None,
        header_gamma: Optional[fits.Header] = None,
        index_data: int = 0,
        index_beam: int = 0,
        index_gamma: int = 0,
    ) -> Tokult:
        '''Constructer of ``Tokult``.

        Construct an instance easier than to use the init constructer.

        Args:
            data (Union[np.ndarray, str]): Observed data. The format is a data array or
                a fits file name.
            beam (Union[np.ndarray, str, None], optional): Dirty beam or point spread
                function (PSF). The format is a data array or a fits file name.
                Defaults to None.
            gravlens (Union[tuple[np.ndarray, ...], tuple[str, ...]], None, optional):
                Gravitational lensing parameters. The format is a tuple containing
                the three data array or fits file names of parameters: gamma1, gamma2,
                and kappa. Defaults to None.
            header_data (Optional[fits.Header], optional): Header of data.
                Defaults to None.
                This is necessary when the ``data`` is not given in a fits file.
            header_beam (Optional[fits.Header], optional): Header of psf.
                Defaults to None.
                This is necessary when ``beam`` is not given in a fits file.
            header_gamma (Optional[fits.Header], optional): Header of gamma.
                Defaults to None.
                This is necessary when ``gravlens`` is not given in fits files.
            index_data (int, optional): Index of fits extensions of the data fits file.
                Defaults to 0.
            index_beam (int, optional): Index of fits extensions of the beam fits file.
                Defaults to 0.
            index_gamma (int, optional): Index of fits extensions of the lens fits
                files. Defaults to 0.

        Examples:
            >>> tok = tokult.Tokult.launch('data.fits', 'psf.fits',
                                           ('gamma1.fits', 'gamma2.fits', 'kappa.fits'))
        '''
        datacube = DataCube.create(data, header=header_data, index_hdul=index_data)

        dirtybeam: Optional[DirtyBeam]
        if beam is not None:
            dirtybeam = DirtyBeam.create(
                beam, header=header_beam, index_hdul=index_beam
            )
        else:
            dirtybeam = None

        gl: Optional[GravLens]
        if gravlens is not None:
            g1, g2, k = gravlens
            gl = GravLens.create(g1, g2, k, header=header_gamma, index_hdul=index_gamma)
            gl.match_wcs_with(datacube)
        else:
            gl = None
        return cls(datacube, dirtybeam, gl)

    def imagefit(
        self,
        init: Sequence[float],
        bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
        fix: Optional[fitting.FixParams] = None,
        optimization: str = 'mc',
        niter: int = 1,
        nperturb: int = 1000,
        nwalkers: int = 64,
        nsteps: int = 5000,
        pool: Optional[Pool] = None,
        is_separate: bool = False,
        mask_for_fit: Optional[np.ndarray] = None,
        progressbar: bool = False,
    ) -> fitting.Solution:
        '''Fit a 3d model to the data cube on the image plane.

        Args:
            init (Sequence[float]): Initial parameters of fitting.
            bound (Optional[tuple[Sequence[float], Sequence[float]], optional):
                Boundaries of parameters. Defaults to None.
                When None is given, the default parameter boundaries are used.
                The boundaries can be easily set using ``get_bound_params``.
                Currently, in the mcmc method, only flat prior distributions
                are available through this argument.
            fix (Optional[fitting.FixParams], optional): Defaults to None.
            optimization (str, optional): Defaults to 'mc'.
            niter (int, optional): Number of iterations of fitting, used
                in the least square method. Defaults to 1.
            nperturb (int, optional): Number of perturbations in the Monte Carlo
                method. Defaults to 1000.
            nwalkers (int, optional): Number of walkers, used in the MCMC method.
                Defaults to 64.
            nsteps (int, optional): Number of steps, used in the MCMC method.
                Defaults to 5000.
            pool (Optional[Pool], optional): multiprocessing.pool for a multi-process
                MCMC fitting. Defaults to None.
            is_separate (bool, optional): If True, parameters regarding kinematics
                and emission are separated; and thus fitting uses all the 14
                parameters. If False, the parameters are the same and
                the number of free parameters are reduced. Defaults to False.
            mask_for_fit (Optional[np.ndarray], optional): Mask specifying pixels
                used for fitting. Defaults to None.
            progressbar (bool, optional): If True, a progress bar is shown.
                Defaults to False.

        Returns:
            fitting.Solution: Fitting results and related parameters.

        Examples:
            >>> sol = tok.imagefit(init, bound, optimization='mc')
        '''
        func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
        func_lensing = self.gravlens.lensing if self.gravlens else None

        if optimization == 'mcmc':
            solution = fitting.mcmc(
                self.datacube,
                init,
                bound,
                func_convolve=func_convolve,
                func_lensing=func_lensing,
                mode_fit='image',
                fix=fix,
                is_separate=is_separate,
                mask_for_fit=mask_for_fit,
                nwalkers=nwalkers,
                nsteps=nsteps,
                pool=pool,
                progressbar=progressbar,
            )
        elif optimization == 'ls':
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
                mask_for_fit=mask_for_fit,
            )
        elif optimization == 'mc':
            func_fullconvolve = self.dirtybeam.fullconvolve if self.dirtybeam else None
            solution = fitting.montecarlo(
                self.datacube,
                init,
                bound,
                func_convolve=func_convolve,
                func_fullconvolve=func_fullconvolve,
                func_lensing=func_lensing,
                niter=niter,
                nperturb=nperturb,
                fix=fix,
                is_separate=is_separate,
                mask_for_fit=mask_for_fit,
                progressbar=progressbar,
            )
        _redshift_tmp = self.gravlens.z_source if self.gravlens is not None else None
        solution.set_metainfo(z=_redshift_tmp, header=self.datacube.header)
        self.construct_modelcube(solution.best)
        return solution

    def uvfit(
        self,
        init: Sequence[float],
        bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
        fix: Optional[fitting.FixParams] = None,
        optimization: str = 'mcmc',
        niter: int = 1,
        nwalkers: int = 64,
        nsteps: int = 5000,
        pool: Optional[Pool] = None,
        is_separate: bool = False,
        mask_for_fit: Optional[np.ndarray] = None,
        progressbar: bool = False,
    ) -> fitting.Solution:
        '''Fit a 3d model to the data cube on the uv plane.

        Args:
            init (Sequence[float]): Initial parameters of fitting.
                a function to guess initial parameters: ``tokult.initialguess()``.
            bound (Optional[tuple[Sequence[float], Sequence[float]], optional):
                Boundaries of parameters. Defaults to None.
                When None is given, the default parameter boundaries are used.
                The boundaries can be easily set using ``get_bound_params``.
                Currently, in the mcmc method, only flat prior distributions
                are available through this argument.
            fix (Optional[fitting.FixParams], optional): Fix parameters during
                fitting. See ``FixParams``. Defaults to None.
            optimization (str, optional): Method to optimize the 3D model.
                - 'ls': least square method.
                - 'mcmc': Malcov Chain Monte Carlo method.
                Defaults to 'mcmc'.
            niter (int, optional): Number of iterations of fitting, used
                in the least square method. Defaults to 1.
            nperturb (int, optional): Number of perturbations in the Monte Carlo
                method. Defaults to 1000.
            nwalkers (int, optional): Number of walkers, used in the MCMC method.
                Defaults to 64.
            nsteps (int, optional): Number of steps, used in the MCMC method.
                Defaults to 5000.
            pool (Optional[Pool], optional): multiprocessing.pool for a multi-process
                MCMC fitting. Defaults to None.
            is_separate (bool, optional): If True, parameters regarding kinematics
                and emission are separated; and thus fitting uses all the 14
                parameters. If False, the parameters are the same and
                the number of free parameters are reduced. Defaults to False.
            mask_for_fit (Optional[np.ndarray], optional): Mask specifying pixels
                used for fitting. In uv fitting, specifying the uv-coverage is
                important, which is passed through this argument. Defaults to None.
            progressbar (bool, optional): If True, a progress bar is shown.
                Defaults to False.

        Returns:
            fitting.Solution: Fitting results and related parameters.

        Examples:
            >>> sol = tok.uvfit(init, bound, optimization='mcmc')

        Note:
            The input parameters are the same as ``imagefit``.
        '''
        if self.dirtybeam:
            beam_visibility = self.dirtybeam.uvplane
            norm_weight = self.calculate_normweight()
        else:
            msg = '"DirtyBeam" is necessary for uvfit.'
            c.logger.warning(msg)
            raise ValueError(msg)
        func_lensing = self.gravlens.lensing if self.gravlens else None

        if optimization == 'mcmc':
            solution = fitting.mcmc(
                self.datacube,
                init,
                bound,
                beam_vis=beam_visibility,
                norm_weight=norm_weight,
                func_lensing=func_lensing,
                mode_fit='uv',
                fix=fix,
                is_separate=is_separate,
                mask_for_fit=mask_for_fit,
                nwalkers=nwalkers,
                nsteps=nsteps,
                pool=pool,
                progressbar=progressbar,
            )
        elif optimization == 'ls':
            solution = fitting.least_square(
                self.datacube,
                init,
                bound,
                beam_vis=beam_visibility,
                norm_weight=norm_weight,
                func_lensing=func_lensing,
                mode_fit='uv',
                fix=fix,
                is_separate=is_separate,
                mask_for_fit=mask_for_fit,
                niter=niter,
            )
        _redshift_tmp = self.gravlens.z_source if self.gravlens is not None else None
        solution.set_metainfo(z=_redshift_tmp, header=self.datacube.header)
        self.construct_modelcube(solution.best)
        return solution

    def initialguess(self, is_separate: bool = False) -> fitting.InputParams:
        '''Guess initial input parameters for fitting.

        Fit tow-dimensional moment-0 (flux) map and moment-1 (velocity) map then
        Estimate input parameters from the 2-d fitting results.

        Args:
            is_separate (bool, optional): If True, parameters regarding kinematics
                and emission are separated; and thus fitting uses all the 14
                parameters. If False, the parameters are the same and
                the number of free parameters are reduced. Defaults to False.

        Returns:
            fitting.InputParams: Best-guessed input parameters.

        Examples:
            >>> init = tok.initialguess()

        Note:
            Initial parameters are crucial for parameter fitting, especially in
            the least-square and Monte Carlo methods and being related with speed
            of convergence in the MCMC method.
        '''
        func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
        func_lensing = self.gravlens.lensing if self.gravlens else None
        return fitting.initialguess(
            self.datacube, func_convolve, func_lensing, is_separate
        )

    def use_region(
        self,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        '''Use a region of datacube used for fitting.

        Args:
            xlim (Optional[tuple[int, int]], optional): The limit of the x-axis.
                Defaults to None.
            ylim (Optional[tuple[int, int]], optional): The limit of the y-axis.
                Defaults to None.
            vlim (Optional[tuple[int, int]], optional): The limit of the v-axis.
                Defaults to None.

        Returns:
            None:

        Examples:
            >>> tok.use_region((32, 96), (32, 96), (5, 12))

        Note:
            In ``uvfit``, the v-axis limit must be specified smaller than
            original cube size, because ``uvfit`` estimates the noise level
            using the pixels outside ``vlim``.
        '''
        self.datacube.cutout(xlim, ylim, vlim)
        if self.dirtybeam is not None:
            self.dirtybeam.cutout_to_match_with(self.datacube)
        if self.gravlens is not None:
            self.gravlens.match_wcs_with(self.datacube)

    def use_redshifts(
        self, z_source: float, z_lens: float, z_assumed: float = np.inf
    ) -> None:
        '''Set the redshifts of the source and the lens galaxies.

        The redshifts are used to compute the gravitational lensing effects and
        to convert the parameters to the physical units.

        Args:
            z_source (float): The source (galaxy) redshift.
            z_lens (float): The lens (cluster) redshift.
            z_assumed (float, optional): The redshift assumed in the
                gravitational parameters. If D_s / D_L = 1, the value
                should be infinite (``np.inf``). Defaults to ``np.inf``.

        Returns:
            None:

        Examples:
            >>> tok.use_redshift(z_source=6.2, z_lens=0.9)
        '''
        if self.gravlens is not None:
            self.gravlens.use_redshifts(z_source, z_lens, z_assumed)

    def change_datacube(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        '''Change the variable ``datacube``.

        May be useful to change ``datacube`` of an instance.
        '''
        self.datacube = DataCube.create(
            data_or_fname, header=header, index_hdul=index_hdul
        )
        self.use_region(xlim, ylim, vlim)

    def change_dirtybeam(
        self,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        '''Change the variable ``dirtybeam``.

        May be useful to change ``dirtybeam`` of an instance.
        '''
        self.dirtybeam = DirtyBeam.create(
            data_or_fname, header=header, index_hdul=index_hdul
        )

    def change_gravlens(
        self,
        data_or_fname: Union[tuple[np.ndarray, ...], tuple[str, ...]],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> None:
        '''Change the variable ``gravlens``.

        May be useful to change ``gravlens`` of an instance.
        '''
        g1, g2, k = data_or_fname
        self.gravlens = GravLens.create(g1, g2, k, header=header, index_hdul=index_hdul)

    def construct_modelcube(self, params: tuple[float, ...]) -> None:
        '''Construct ``modelcube`` from the input parameters.

        Construct a 3D model and set it to an internal variable ``modelcube``.
        If you want to use a model outside ``Tokult`` instances, please use
        ``ModelCube.create()`` instead.

        Args:
            params (tuple[float, ...]): Input parameters.

        Returns:
            None:

        Examples:
            Change a part of the best-fit parameters and construct the model.

            >>> params = sol.best._replace(PA_dyn=0.0)
            >>> tok.construct_modelcube(params)

        Note:
            This method needs the global parameters to be set already.
            You may need to fit the data once, before using this method.
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

    def calculate_normweight(self) -> float:
        '''Calculate the normalization weight used in ``uvfit``.

        The obtained value is almost equal to sum-of-weight, but different by a
        factor of a few.

        Returns:
            float: The normalization weight
        '''
        assert self.dirtybeam is not None
        uv = self.datacube.rfft2(self.datacube.original)
        uvpsf = misc.rfft2(self.dirtybeam.original)
        uv_noise = uv / np.sqrt(abs(uvpsf.real))

        v0, v1 = self.datacube.vlim
        n = uv_noise[[v0 - 1, v1], :, :].real
        p = uvpsf[[v0 - 1, v1], :, :].real
        return 1.0 / n[p > -p.min()].std() ** 2


class Cube(object):
    '''3D data cube.

    Examples:
        >>>

    Attributes:
        imageplane (np.ndarray): Cutout 3D data cube on the image plane.
        uvplane (np.ndarray): Cutout 3D data cube on the uv plane.
            This is the Fourier transformation of ``imageplane``.
        original (np.ndarray): Original-size, 3D data cube.
        header (Optional[fits.Header]): Header of the fits data.
            Defaults to None.
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
        self.uvplane = self.rfft2(self.original, zero_padding=True)
        self.mask_FoV = np.logical_not(np.equal(self.original, 0.0)).astype(int)

        self.xlim: tuple[int, int]
        self.ylim: tuple[int, int]
        self.vlim: tuple[int, int]
        self.xslice: slice
        self.yslice: slice
        self.vslice: slice
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
        '''Cutout 3D cube from ``original``.

        Args:
            xlim (Optional[tuple[int, int]], optional): The limit of the x-axis.
                Defaults to None.
            ylim (Optional[tuple[int, int]], optional): The limit of the y-axis.
                Defaults to None.
            vlim (Optional[tuple[int, int]], optional): The limit of the v-axis.
                Defaults to None.

        Returns:
            None:

        Examples:
            >>> cube.cutout((32, 96), (32, 96), (5, 12))
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
        self.uvplane = self.rfft2(self.original[self.vslice, :, :], zero_padding=True)

    def rms(self, is_originalsize: bool = False) -> np.ndarray:
        '''Compute the rms noise of the data cube.

        Compute the rms noise using pixels outside of the region used for
        ``imageplane``, by assuming that the pixels are not affected by
        any objects and reflect pure noises.

        Args:
            is_originalsize (bool, optional): If False, the computed rms noise is
            limited at ``vlim`` of ``imageplane``. If True, the rms noise is
            computed using the original-size data cube. Defaults to False.

        Returns:
            np.ndarray: the one-dimensional array containing the rms noises at
            each pixel (channel) along the velocity axis.

        Examples:
            >>> rms = cube.rms()

        Note:
            This method may not return the correct rms if multiple objects are
            detected in the Field of View.
        '''
        if is_originalsize:
            image = self.original
        else:
            image = self.original[self.vslice, :, :]
        maskedimage = np.copy(image)
        maskedimage[:, self.yslice, self.xslice] = 0.0
        rms = misc.rms(maskedimage, axis=(1, 2))
        assert isinstance(rms, np.ndarray)
        return rms

    def moment0(self, is_originalsize: bool = False) -> np.ndarray:
        '''Moment-0 (integrated-flux) map.

        The moment-0 map is the flux map integrated along the velocity axis.
        The default computed area is the one defined by ``Cube.cutout()``.

        Args:
            is_originalsize (bool, optional): If False, compute the moment-0 map
                using the cutout region. If True, using the original-size data.
                Defaults to False.

        Returns:
            np.ndarray: Two-dimensional moment-0 map.

        Examples:
            >>> mom0 = cube.moment0()
        '''
        if is_originalsize:
            return np.sum(self.original, axis=0)
        else:
            return np.sum(self.imageplane, axis=0)

    def rms_moment0(self, is_originalsize: bool = False) -> float:
        '''RMS noise of the moment 0 map.

        RMS noise is computed using the region outside the cutout region, where
        the object is located.

        Args:
            is_originalsize (bool, optional): If False, the moment-0 map is
                computed using the cutout region. If True, using the original-
                size data. Defaults to False.

        Returns:
            float: rms of the moment-0 map.
        '''
        if is_originalsize:
            image = self.original
        else:
            image = self.original[self.vslice, :, :]
        maskedimage = np.sum(image, axis=0)
        maskedimage[self.yslice, self.xslice] = 0.0
        rms = misc.rms(maskedimage)
        assert isinstance(rms, float)
        return rms

    def pixmoment1(self, thresh: float = 0.0) -> np.ndarray:
        '''Moment-1 (velocity) map.

        Args:
            thresh (float, optional): Threshold of the pixel values on the moment-
            0 map. In the pixels below this threshold, ``np.nan`` is inserted.
            Defaults to 0.0.

        Returns:
            np.ndarray: Two-dimensional moment-1 map.

        Examples:
            To output pixels whose moment-0 values are two times higher than the
            rms noise of the moment-0 map.

            >>> mom1 = cube.pixmoment1(thresh=2 * cube.rms_moment0())

        Note:
            The units of the moment-1 map is *pixel*. You may change the units
            by multiplying the results by the velocity-bin width.
        '''
        mom0 = self.moment0()
        mom1 = np.sum(self.imageplane * self.vgrid, axis=0) / mom0
        mom1[mom0 <= thresh] = np.nan
        return mom1

    def pixmoment2(self, thresh: float = 0.0) -> np.ndarray:
        '''Moment-2 (velocity-dispersion) map.

        Args:
            thresh (float, optional): Threshold of the pixel values on the moment-
            0 map. In the pixels below this threshold, ``np.nan`` is inserted.
            Defaults to 0.0.

        Returns:
            np.ndarray: Two-dimensional moment-2 map.

        Examples:
            To output pixels whose moment-0 values are two times higher than the
            rms noise of the moment-0 map.

            >>> mom2 = cube.pixmoment2(thresh=2 * cube.rms_moment0())

        Note:
            The units of the moment-2 map is *pixel*. You may change the units
            by multiplying the results by the velocity-bin width.
        '''
        mom0 = self.moment0()
        mom1 = self.pixmoment1()
        vv = self.vgrid - mom1[np.newaxis, ...]
        mom2 = np.sum(self.imageplane * np.sqrt(vv ** 2), axis=0) / mom0
        mom2[mom0 <= thresh] = None
        return mom2

    def _get_pixmoments(
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
            mom0 = self._get_pixmoments(imom=0)
            self.mom1 = np.sum(self.imageplane * self.vgrid, axis=0) / mom0
            self.mom1[mom0 <= thresh] = None
            return self.mom1
        if imom == 2:
            if not recalc:
                try:
                    return self.mom2
                except AttributeError:
                    pass
            mom1 = self._get_pixmoments(imom=1)
            vv = self.vgrid - mom1[np.newaxis, ...]
            self.mom2 = np.sum(self.imageplane * np.sqrt(vv ** 2), axis=0) / self.mom0
            self.mom2[self.mom0 <= thresh] = None
            return self.mom2

        message = 'An input "imom" should be the int type of 0, 1, or 2.'
        c.logger.error(message)
        raise ValueError(message)

    def noisy(
        self,
        rms: Union[float, np.ndarray],
        convolve: Optional[Callable] = None,
        seed: Optional[int] = None,
        is_originalsize: bool = False,
        uvcoverage: Optional[np.ndarray] = None,
    ):
        '''Create a noisy mock data cube.

        The noisy cube is created by adding noise to the contained 3D data cube.
        This means that ``convolve`` should be the same as applied to the
        contained data cube. This method is useful to perturb the data cube for
        the Monte Carlo estiamtes of fitting errors.

        Args:
            rms (Union[float, np.ndarray]): RMS of the added noise cube. The rms
                is computed at each pixel (channel) along the velocity axis.
            convolve (Optional[Callable], optional): Convolution function.
                Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to None.
            is_originalsize (bool, optional): If False, the size of the noisy cube
                the same as the cutout ``imageplane``. If True, the cube is the
                original size. Defaults to False.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane. The
                pixels with False are set to 0.0. Defaults to None.
        '''
        noise = self.create_noise(rms, self.original.shape, convolve, seed, uvcoverage)
        mock = self.original + noise
        if not is_originalsize:
            mock = mock[self.vslice, self.yslice, self.xslice]
        return mock

    @staticmethod
    def rfft2(data: np.ndarray, zero_padding: bool = False) -> np.ndarray:
        '''Wrapper of misc.rfft2.

        This method add the new argument ``zero_padding`` for observed data.

        Args:
            data (np.ndarray): Data cube on the image plane.
            zero_padding (bool, optional): If True, zero-padding the pixels with
                ``None``. Defaults to False.

        Returns:
            np.ndarray: Fourier-transformed data cube on the uv plane.

        Examples:
            >>> uv = cube.rfft2(image, zero_padding=True)
        '''
        if np.any(idx := (np.logical_not(np.isfinite(data)))):
            if zero_padding:
                data[idx] = 0.0
            else:
                raise ValueError(
                    'Input cube data includes non-finite values (NaN or Inf).'
                )
        return misc.rfft2(data)

    @staticmethod
    def create_noise(
        rms: Union[float, np.ndarray],
        shape: tuple[int, ...],
        convolve: Optional[Callable] = None,
        seed: Optional[int] = None,
        uvcoverage: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        '''Create a noise cube.

        Args:
            rms (Union[float, np.ndarray]): RMS noise of the cube.
            shape (tuple[int, ...]): Shape of the cube.
            convolve (Optional[Callable], optional): Convolution function.
                Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to None.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane.
                Defaults to None.

        Returns:
            np.ndarray: Noise cube.

        Examples:
            >>> noise = cube.create_noise(rms, shape, func_convlution)
        '''
        rng = default_rng(seed)
        noise = rng.standard_normal(size=shape)
        if convolve:
            noise = convolve(noise, uvcoverage=uvcoverage, is_noise=True)
        rms_computed = misc.rms(noise)
        noise *= rms / rms_computed
        return noise


class DataCube(Cube):
    '''Cube class to contain an observed datacube.
    '''

    def perturbed(
        self,
        convolve: Optional[Callable] = None,
        seed: Optional[int] = None,
        is_originalsize: bool = False,
        uvcoverage: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        '''Perturb the data cube with the same noise level and return it.

        This method is useful to perturb the data cube for the Monte Carlo
        estiamtes of fitting errors.

        Args:
            convolve (Optional[Callable], optional): Convolution function.
                Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to None.
            is_originalsize (bool, optional): If False, the output is the
                perturbed ``imagepalne``. If True, the perturbed ``original``
                Defaults to False.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane.
                Defaults to None.

        Returns:
            np.ndarray: Perturbed data cube.

        Examples:
            >>> cube_perturbed = datacube.perturbed(convolve=func_fullconvolve)
        '''
        rms = self.rms(is_originalsize=True)
        rms = rms[..., np.newaxis, np.newaxis]
        return self.noisy(rms, convolve, seed, is_originalsize, uvcoverage)

    @classmethod
    def create(
        cls,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> DataCube:
        '''Constructer of ``DataCube``.

        Args:
            data_or_fname (Union[np.ndarray, str]): Data array or fits file name.
            header (Optional[fits.Header], optional): Header of the fits file,
                necessary if ``data_or_fname`` is a data array. Defaults to None.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.

        Returns:
            DataCube: Data cube.

        Examples:
            >>> datacube = DataCube.create('data.fits')

        Note:
            When the file name is give, the loaded data is squeezed; that is, the
            axis with zero size is dropped. Specifically, the polarization axis of
            the ALMA fits data may be dropped.
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
        '''Read a data cube from a fits file.

        Args:
            fname (str): Fits file name.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.

        Returns:
            tuple[np.ndarray, fits.Header]: Tuple of the data and the fits header.

        Note:
            When the file name is give, the loaded data is squeezed; that is, the
            axis with zero size is dropped. Specifically, the polarization axis of
            the ALMA fits data may be dropped.
        '''
        with fits.open(fname) as hdul:
            # np.squeeze is needed to erase the polari axis.
            imageplane = np.squeeze(hdul[index_hdul].data)
            header = hdul[index_hdul].header
        return imageplane, header


class ModelCube(Cube):
    '''Cube class to contain a modeled datacube.

    This class is especially for the best-fit model cube.
    '''

    def __init__(
        self,
        imageplane: np.ndarray,
        raw: Optional[np.ndarray] = None,
        xlim: Optional[tuple[int, int]] = None,
        ylim: Optional[tuple[int, int]] = None,
        vlim: Optional[tuple[int, int]] = None,
    ) -> None:
        super().__init__(imageplane, xlim=xlim, ylim=ylim, vlim=vlim)
        self.raw = raw

    @classmethod
    def create(
        cls,
        params: tuple[float, ...],
        datacube: DataCube,
        lensing: Optional[Callable] = None,
        convolve: Optional[Callable] = None,
    ) -> ModelCube:
        '''Constructer of ``ModelCube``.

        Args:
            params (tuple[float, ...]): Input parameters.
            datacube (DataCube): Data cube. The size of the model cube is based on
                this data cube.
            lensing (Optional[Callable], optional): Lensing function.
                Defaults to None.
            convolve (Optional[Callable], optional): Convolving function.
                Defaults to None.

        Returns:
            ModelCube: Model cube.

        Examples:
            >>> model = ModelCube.create(params, tok.datacube,
                                         lensing=tok.gravlens.lensing,
                                         convolve=tok.dirtybeam.fullconvolve)
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
            # model_convolved = np.empty_like(modelcube)
            # for i, image in enumerate(modelcube):
            model_convolved = convolve(modelcube)
        else:
            model_convolved = modelcube

        model_masked = model_convolved * datacube.mask_FoV
        xlim, ylim, vlim = datacube.xlim, datacube.ylim, datacube.vlim
        return cls(model_masked, raw=modelcube, xlim=xlim, ylim=ylim, vlim=vlim)

    def to_mockcube(
        self,
        rms: Union[float, np.ndarray],
        convolve: Optional[Callable] = None,
        seed: Optional[int] = None,
        is_originalsize: bool = False,
        uvcoverage: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        '''Convert to and output noisy mock cube.

        Args:
            rms (Union[float, np.ndarray]): RMS noise added to the model cube,
                after convolution.
            convolve (Optional[Callable], optional): Convolution function. This
                convolve both *raw* model cube and noise. Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to None.
            is_originalsize (bool, optional): If False, the output mock cube is
                the same size as ``imageplane``. If True, the size is the same
                as ``original``. Defaults to False.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane.
                Defaults to None.

        Returns:
            np.ndarray: Mock data cube.

        Examples:
            >>> mock = modelcube.to_mockcube(
                           tok.datacube.rms(),
                           tok.dirtybeam.fullconvolve,
                           is_original=True)
        '''
        if self.raw is None:
            raise ValueError('Raw model is None.')

        model = convolve(self.raw, uvcoverage=uvcoverage) if convolve else self.raw
        noise = self.create_noise(rms, self.raw.shape, convolve, seed, uvcoverage)
        mock = model + noise
        if not is_originalsize:
            mock = mock[self.vslice, self.yslice, self.xslice]
        return mock


class DirtyBeam:
    '''Contains dirtybeam, i.e., Point Spread Functions (PSF) of Cube.

    Contained data is dirtybeam images as a function of frequency.
    '''

    def __init__(self, beam: np.ndarray, header: Optional[fits.Header] = None,) -> None:
        self.original = beam
        self.imageplane = beam
        self.header = header
        self.uvplane = misc.rfft2(self.original)

    @classmethod
    def create(
        cls,
        data_or_fname: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> DirtyBeam:
        '''Constructer of ``DirtyBeam``.

        Args:
            data_or_fname (Union[np.ndarray, str]): Data array or fits file name.
            header (Optional[fits.Header], optional): Header of the fits file.
                Defaults to None.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.

        Returns:
            DirtyBeam: instance of ``DirtyBeam``.
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

    def convolve(self, image: np.ndarray) -> np.ndarray:
        '''Convolve ``imageplane`` with dirtybeam (psf) in two dimension.

        Perform two-dimensional convolution at each pixel (channel) along the
        velocity axis.

        Args:
            image (np.ndarray): Image to be convolved. Note that the size must be
                the same as the attribute ``imageplane``.

        Returns:
            np.ndarray: 2D-convolved cube.

        Examples:
            >>> convolved_image = dirtybeam.convole(image)
        '''
        # s1 = np.arange(c.conf.kernel_num)
        # t1 = np.arange(c.conf.kernel_num)
        # s2, t2 = np.meshgrid(s1, t1)
        # s3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + s2
        # t3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + t2
        # st = np.array(c.conf.num_pix * s3 + t3, dtype=int)
        # kernel = self.beam[st]
        # kernel2 = kernel / np.sum(kernel)
        kernel = self.imageplane
        # kernel = beam  / np.sum(beam)
        dim = len(image.shape)
        if dim == 2:
            return misc.fftconvolve(image[np.newaxis, :, :], kernel[[0], :, :])
        elif dim == 3:
            return misc.fftconvolve(image, kernel)
        else:
            raise ValueError(f'dimension of image is two or three, not {dim}.')

    def fullconvolve(
        self,
        image: np.ndarray,
        uvcoverage: Optional[np.ndarray] = None,
        is_noise: bool = False,
    ) -> np.ndarray:
        '''Convolve ``original`` with dirtybeam (psf) in two dimension.

        Difference between ``convolve()`` and ``fullconvolve()`` is the size of
        the input ``image``. This method ``fullconvolve`` treat the image with the
        same size as the ``dirtybeam.original``.

        Args:
            image (np.ndarray): Image to be convolved. Note that the size must be
                the same as the attribute ``original``.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane.
                Defaults to None.
            is_noise (bool, optional): True if ``image`` is data. False if noise.
                Defaults to False.

        Returns:
            np.ndarray: 2D-convolved cube.

        Examples:
            >>> convolved_image = dirtybeam.fullconvole(tok.modelcube.raw)

            How to create convolved noise.

            >>> rng = numpy.random.default_rng()
            >>> noise = rng.standard_normal(size=shape)
            >>> noise = dirtybeam.fullconvolve(
                            noise, uvcoverage=uvcoverage, is_noise=True)
        '''
        kernel = self.original
        dim = len(image.shape)
        if len(image.shape) == 2:
            if is_noise:
                return misc.fftconvolve_noise(
                    image[np.newaxis, :, :], kernel[[0], :, :], uvcoverage
                )
            else:
                return misc.fftconvolve(
                    image[np.newaxis, :, :], kernel[[0], :, :], uvcoverage
                )
        elif len(image.shape) == 3:
            if is_noise:
                return misc.fftconvolve_noise(image, kernel, uvcoverage)
            else:
                return misc.fftconvolve(image, kernel, uvcoverage)
        else:
            raise ValueError(f'dimension of image is two or three, not {dim}.')

    def cutout(
        self,
        xlim: Union[tuple[int, int], slice],
        ylim: Union[tuple[int, int], slice],
        vlim: Union[tuple[int, int], slice],
    ) -> None:
        '''Cutout a cubic region from the dirty beam cube.

        Args:
            xlim (Union[tuple[int, int], slice]): The limit of the x-axis.
            ylim (Union[tuple[int, int], slice]): The limit of the y-axis.
            vlim (Union[tuple[int, int], slice]): The limit of the v-axis.

        Returns:
            None:
        '''
        xslice = slice(*xlim) if isinstance(xlim, tuple) else xlim
        yslice = slice(*ylim) if isinstance(ylim, tuple) else ylim
        vslice = slice(*vlim) if isinstance(vlim, tuple) else vlim
        self.imageplane = self.original[vslice, yslice, xslice]
        self.uvplane = misc.rfft2(self.original[vslice, :, :])

    def cutout_to_match_with(self, cube: DataCube) -> None:
        '''Cutout a region with the same size of the input ``cube``.

        Args:
            cube (DataCube): Datacube. The size of ``cube.original`` must be the
                same as the ``dirtybeam.original``. This method makes the size of
                the ``dirtybeam.imageplane`` the same as ``cube.imageplane``.

        Returns:
            None:
        '''
        _, ysize, xsize = self.original.shape
        xlen = cube.xlim[1] - cube.xlim[0]
        ylen = cube.ylim[1] - cube.ylim[0]
        xslice = self._get_slice_at_center(xsize, xlen)
        yslice = self._get_slice_at_center(ysize, ylen)
        vslice = cube.vslice
        self.cutout(xslice, yslice, vslice)

    @staticmethod
    def loadfits(fname: str, index_hdul: int = 0) -> tuple[np.ndarray, fits.Header]:
        '''Read the dirty beam from a fits file.

        Args:
            fname (str): Fits file name.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.

        Returns:
            tuple[np.ndarray, fits.Header]: Tuple of the data and the fits header.

        CAUTION:
            The data shape of ``DirtyBeam`` must be the same as the ``Datacube``.
            This requirement is naturaly satisfied if the input dirty-beam and
            the image-cube fits files are simulationsly created with CASA.
        '''
        with fits.open(fname) as hdul:
            beam = hdul[index_hdul].data
            header = hdul[index_hdul].header
        return np.squeeze(beam), header

    @staticmethod
    def _get_slice_at_center(len_original: int, len_sub: int) -> slice:
        '''Get a central slice of the original array.

        This submethod aims to give a slice whose center is always identical to
        the center of the cube (i.e, dirty beam).

        Args:
            len_original (int): Length of the data cube.
            len_sub (int): Length of the cutout image.

        Returns:
            slice: Slice around the center.
        '''
        if len_original % 2 == 0:  # even
            margin_end = (len_original - len_sub) // 2
            margin_begin = len_original - len_sub - margin_end
        else:  # odd
            margin_begin = (len_original - len_sub) // 2
        return slice(margin_begin, margin_begin + len_sub)


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

        self.gamma1_cutout = gamma1
        self.gamma2_cutout = gamma2
        self.kappa_cutout = kappa

        self.gamma1 = gamma1
        self.gamma2 = gamma2
        self.kappa = kappa

        self.header = header
        self.jacob = self.get_jacob()

        self.z_lens: Optional[float] = None
        self.z_source: Optional[float] = None
        self.z_assumed: Optional[float] = None
        self.distance_ratio = 1.0

    @classmethod
    def create(
        cls,
        data_or_fname_gamma1: Union[np.ndarray, str],
        data_or_fname_gamma2: Union[np.ndarray, str],
        data_or_fname_kappa: Union[np.ndarray, str],
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
    ) -> GravLens:
        '''Constructer of ``GravLens``.

        Args:
            data_or_fname_gamma1 (Union[np.ndarray, str]): Data array or fits file
                name of a lensing parmeter, gamma1.
            data_or_fname_gamma2 (Union[np.ndarray, str]): Data array or fits file
                name of a lensing parmeter, gamma2.
            data_or_fname_kappa (Union[np.ndarray, str]): Data array or fits file
                name of a lensing parmeter, kappa.
            header (Optional[fits.Header], optional): Header of the fits file.
                This method assumes that lensing parameter maps, gamma1, gamma2,
                and kappa, have the same size and coordinates. Defaults to None.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.

        Returns:
            GravLens: Instance of ``GravLens``.
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

        Args:
            coordinates (np.ndarray): Array of the x and y coordinates on the
                image plane. The shape of the array is (n, m, 2), where (n, m) is
                the shape of x (or y) and x and y have been already concatenated,
                so "2" appears.

        Returns:
            np.ndarray: Coordinates on the source plane. The array shape is
                (n, m, 2)

        Examples:
            >>> coord_image = np.moveaxis(np.array([xx, yy]), 0, -1)
            >>> coord_source = lensing(coord_image)
        '''
        return np.squeeze(self.jacob @ coordinates[..., np.newaxis], -1)

    def get_jacob(self) -> np.ndarray:
        '''Get Jacobian of the lensing equation.

        Returns:
            np.ndarray: Jacobian
        '''
        g1 = self.gamma1
        g2 = self.gamma2
        k = self.kappa

        jacob = np.array([[1 - k - g1, -g2], [-g2, 1 - k + g1]])
        axis = np.concatenate((2 + np.arange(g1.ndim), (0, 1)))
        # assert axis == np.array([2, 3, 0, 1])
        jacob = jacob.transpose(axis)
        # assert jacob.shape == (n, m, 2, 2)
        return jacob

    def match_wcs_with(self, cube: DataCube) -> None:
        '''Match the world coordinate system with the input data cube.

        Use the wcs of ``cube.imageplane``; therefore, the matched lensing
        parameter map become smaller than the original map.

        Args:
            cube (DataCube): ``DataCube`` including the header information. This
                method uses the wcs included in the header of the data cube and
                the ``GravLens`` instance.

        Returns:
            None:

        Examples:
            >>> gravlens.match_wcs_with(datacube)
        '''
        assert self.header is not None
        wcs_cube = wcs.WCS(cube.header)
        wcs_gl = wcs.WCS(self.header)
        skycoord_wcs, _, _ = wcs_cube.pixel_to_world(
            cube.xgrid[0, :, :].ravel(), cube.ygrid[0, :, :].ravel(), 0, 0
        )
        idx = wcs_gl.world_to_array_index(skycoord_wcs)
        shape = cube.xgrid.shape[1:]
        self.gamma1_cutout = self.original_gamma1[idx].reshape(*shape)
        self.gamma2_cutout = self.original_gamma2[idx].reshape(*shape)
        self.kappa_cutout = self.original_kappa[idx].reshape(*shape)
        self.gamma1 = self.gamma1_cutout * self.distance_ratio
        self.gamma2 = self.gamma2_cutout * self.distance_ratio
        self.kappa = self.kappa_cutout * self.distance_ratio
        self.jacob = self.get_jacob()

    def use_redshifts(
        self, z_source: float, z_lens: float, z_assumed: float = np.inf,
    ) -> None:
        '''Correct the lensing parameters using the redshifts.

        Args:
            z_source (float): The source (galaxy) redshift.
            z_lens (float): The lens (cluster) redshift.
            z_assumed (float, optional): The redshift assumed in the
                gravitational parameters. If D_s / D_L = 1, the value
                should be infinite (``np.inf``). Defaults to ``np.inf``.

        Returns:
            None:
        '''
        self.z_lens = z_lens
        self.z_source = z_source
        self.z_assumed = z_assumed
        self.distance_ratio = self.get_angular_distance_ratio(
            z_source, z_lens, z_assumed
        )
        self.gamma1 = self.gamma1_cutout * self.distance_ratio
        self.gamma2 = self.gamma2_cutout * self.distance_ratio
        self.kappa = self.kappa_cutout * self.distance_ratio
        self.jacob = self.get_jacob()

    def reset_redshifts(self) -> None:
        '''Reset the redshift infomation.
        '''
        self.z_lens = None
        self.z_source = None
        self.z_assumed = None
        self.distance_ratio = 1.0
        self.gamma1 = self.gamma1_cutout
        self.gamma2 = self.gamma2_cutout
        self.kappa = self.kappa_cutout
        self.jacob = self.get_jacob()

    def magnification(self) -> np.ndarray:
        '''Get magnification factor using the lensing parameters.

        Returns:
            np.ndarray: Magnification map.
        '''
        gamma2 = self.gamma1 ** 2 + self.gamma2 ** 2
        return 1 / ((1 - self.kappa) ** 2 - gamma2)

    @staticmethod
    def get_angular_distance_ratio(
        z_source: float, z_lens: float, z_assumed: float = np.inf
    ) -> float:
        '''Angular distance ratio of D_LS to D_S, normalized by assumed D_LS/D_S.

        Lensing parameter maps are distributed using some D_LS/D_S at specific
        redshifts. This method provides a new factor that can be multiplied by
        the lensing parameter maps to correct the redshift dependency.

        Args:
            z_source (float): The source (galaxy) redshift.
            z_lens (float): The lens (cluster) redshift.
            z_assumed (float, optional): The redshift assumed in the
                gravitational parameters. If D_s / D_L = 1, the value
                should be infinite (``np.inf``). Defaults to ``np.inf``.

        Returns:
            float: Angular distance ratio, D_LS/D_S

        Examples:
            >>> distance_ratio = gravlens.get_angular_distance_ratio(6.2, 0.9)
            >>> gamma1_new = gamma1_old * distance_ratio
        '''
        D_S = c.cosmo.angular_diameter_distance(z_source)
        D_LS = c.cosmo.angular_diameter_distance_z1z2(z_lens, z_source)
        if np.isinf(z_assumed):
            return (D_LS / D_S).decompose().value
        D_ratio = D_LS / D_S
        D_S_assumed = c.cosmo.angular_diameter_distance(z_assumed)
        D_LS_assumed = c.cosmo.angular_diameter_distance_z1z2(z_lens, z_assumed)
        D_ratio_assumed = D_LS_assumed / D_S_assumed
        return (D_ratio / D_ratio_assumed).decompose().value

    @staticmethod
    def loadfits(
        fname_gamma1: str, fname_gamma2: str, fname_kappa: str, index_hdul: int = 0,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]:
        '''Read gravlens from fits file.

        Args:
            fname_gamma1 (str): Fits file name of the gamma1 map.
            fname_gamma2 (str): Fits file name of the gamma2 map.
            fname_kappa (str): Fits file name of the kappa map.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Assumes that all the fits files include the lensing parameter maps
                at the same extension index. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray, np.ndarray, fits.Header]: [description]
        '''
        with fits.open(fname_gamma1) as hdul:
            gamma1 = hdul[index_hdul].data
            header = hdul[index_hdul].header
        with fits.open(fname_gamma2) as hdul:
            gamma2 = hdul[index_hdul].data
        with fits.open(fname_kappa) as hdul:
            kappa = hdul[index_hdul].data
        return gamma1, gamma2, kappa, header
