def construct_uvmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.'''
    global cubeshape, yslice, xslice
    model_cutout = construct_model_at_imageplane(params)
    model_image = np.zeros(cubeshape)
    model_image[:, yslice, xslice] = model_cutout
    # model_image = construct_model_at_imageplane(params)
    model_visibility = misc.rfft2(model_image)
    # image = misc.ifft2(model_visibility * beam_visibility)
    # return misc.fft2(image * mask_FoV)
    # return model_visibility * beam_visibility
    return model_visibility


class DirtyBeam:
    '''Contains dirtybeam, i.e., Point Spread Functions (PSF) of Cube.

    Contained data is dirtybeam images as a function of frequency.
    '''

    def __init__(self, beam: np.ndarray, header: Optional[fits.Header] = None) -> None:
        self.original = beam
        self.imageplane = beam
        self.header = header
        self.uvplane = misc.rfft2(self.original)
        # sometimes divide by zero encountered in fitting
        self.uvplane[self.uvplane == 0] = misc.min_abs(self.uvplane)

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
        if dim == 2:
            if is_noise:
                return misc.fftconvolve_noise(
                    image[np.newaxis, :, :], kernel[[0], :, :], uvcoverage
                )
            else:
                return misc.fftconvolve(
                    image[np.newaxis, :, :], kernel[[0], :, :], uvcoverage
                )
        elif dim == 3:
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
        # sometimes divide by zero encountered in fitting
        self.uvplane[self.uvplane == 0] = misc.min_abs(self.uvplane)

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
