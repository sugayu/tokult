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
        mom2 = np.sum(self.imageplane * np.sqrt(vv**2), axis=0) / mom0
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
            self.mom2 = np.sum(self.imageplane * np.sqrt(vv**2), axis=0) / self.mom0
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
        rms_of_standardnoise: Optional[np.ndarray] = None,
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
        noise = self.create_noise(
            rms, self.original.shape, convolve, seed, uvcoverage, rms_of_standardnoise
        )
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
        rms_of_standardnoise: Optional[np.ndarray] = None,
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
        # noise = misc.irfft2(misc.create_uvnoise_standardgauss(size=shape, seed=seed))
        rng = default_rng(seed)
        noise = rng.standard_normal(size=shape)
        if convolve:
            noise = convolve(noise, uvcoverage=uvcoverage, is_noise=True)
        if rms_of_standardnoise is None:
            rms_of_standardnoise = np.asarray(misc.rms(noise, axis=(1, 2)))
            rms_of_standardnoise = rms_of_standardnoise[..., np.newaxis, np.newaxis]
        assert rms_of_standardnoise is not None
        noise *= rms / rms_of_standardnoise
        return noise

    @staticmethod
    def _estimate_rms_of_standardnoise(
        shape: tuple[int, ...],
        convolve: Optional[Callable] = None,
        uvcoverage: Optional[np.ndarray] = None,
    ):
        '''Estimate rms of mock noises after considering uvcoverage.

        This function is for perturbation of a datacube.
        Although the create_noise() generates noises based on the standard normal
        distribution, which has the rms of 1.0, the rms changes after taking the
        uvcoverage into accound. This function estimate the rms value that can
        be used to generate noises in Monte Carlo simulations.
        Note: Without the rms returned by this function, the create_noise()
        generates the noise with the rms that is exactly the same as the input value.

        Args:
            shape (tuple[int, ...]): Shape of the cube.
            convolve (Optional[Callable], optional): Convolution function.
                Defaults to None.
            seed (Optional[int], optional): Random seed. Defaults to None.
            uvcoverage (Optional[np.ndarray], optional): Mask on the uv plane.
                Defaults to None.
        '''
        Niter = 100
        rng = default_rng()
        array_noise = np.empty((Niter, shape[0]))

        for i in range(Niter):
            noise = rng.standard_normal(shape)
            if convolve:
                noise = convolve(noise, uvcoverage=uvcoverage, is_noise=True)
            array_noise[i, :] = np.asarray(misc.rms(noise, axis=(1, 2)))
        return array_noise.mean(axis=0)[..., np.newaxis, np.newaxis]


class DataCube(Cube):
    '''Cube class to contain an observed datacube.'''

    def perturbed(
        self,
        convolve: Optional[Callable] = None,
        seed: Optional[int] = None,
        is_originalsize: bool = False,
        uvcoverage: Optional[np.ndarray] = None,
        rms_of_standardnoise: Optional[np.ndarray] = None,
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
        return self.noisy(
            rms, convolve, seed, is_originalsize, uvcoverage, rms_of_standardnoise
        )

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
        create_interpolate_lensing: Optional[Callable] = None,
        upsampling_rate: tuple[int, ...] = (1, 1, 1),
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
            xx_grid_image=datacube.xgrid,
            yy_grid_image=datacube.ygrid,
            vv_grid_image=datacube.vgrid,
            cubeshape_imageplane=datacube.imageplane.shape,
            lensing=lensing,
            create_interpolate_lensing=create_interpolate_lensing,
            upsampling_rate=upsampling_rate,
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
