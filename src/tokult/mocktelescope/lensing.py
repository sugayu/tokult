'''Gravitational lensing.
'''

import numpy as np
from scipy.interpolate import RectBivariateSpline
from astropy import wcs
from .gridconvert import GridConverter
from logging import getLogger

logger = getLogger(__name__)


class GravLens(GridConverter):
    '''Change coordinates according to a given gravitaitonal lens map.

    This class provides simple converter of the pixel grids. If you want to use misc functions
    to treat lensing effects, use a class XXX instead.

    Attributes:
        lensmap [np.ndarray]: Array including pixel deflection maps with shape of (ny, nx, 2),
            which is the same as the original (input) grids. lensmap[:,:,0] includes x-direction
            and lensmap[:,:,1] includes y-direction.
    '''

    def __init__(self, lensmap: np.ndarray) -> None:
        self.pixel_deflect = lensmap
        super().__init__()

    def convert(self, grids: np.ndarray) -> np.ndarray:
        '''Main method to change the grid coordinate.'''
        newgrids = (
            grids[:, :, 0] - self.pixel_deflect[:, :, 0],  # x
            grids[:, :, 1] - self.pixel_deflect[:, :, 1],  # y
        )
        return np.array(newgrids)


class PreviousGravLens:
    '''Deal with gravitational lensing effects based on a given lens models.

    Contents are lensing parameters depending on positions.
    '''

    def __init__(
        self,
        x_arcsec_deflect: np.ndarray,
        y_arcsec_deflect: np.ndarray,
        header: fits.Header,
        z_source: Optional[float] = None,
        z_lens: Optional[float] = None,
        z_assumed: Optional[float] = None,
    ) -> None:
        self.original_x_arcsec_deflect = x_arcsec_deflect
        self.original_y_arcsec_deflect = y_arcsec_deflect
        # self.idx_wcs = np.isfinite(x_arcsec_deflect)
        # self.shape = x_arcsec_deflect.shape
        self.original_xaxis = np.arange(x_arcsec_deflect.shape[1])
        self.original_yaxis = np.arange(x_arcsec_deflect.shape[0])
        self.xaxis = np.arange(x_arcsec_deflect.shape[1])
        self.yaxis = np.arange(x_arcsec_deflect.shape[0])

        self.x_arcsec_deflect: np.ndarray
        self.y_arcsec_deflect: np.ndarray
        self.x_pixel_deflect: np.ndarray
        self.y_pixel_deflect: np.ndarray
        self.header = header
        self.header_datacube: Optional[fits.Header] = None

        self.interpolate_x_arcsec = RectBivariateSpline(
            self.original_yaxis, self.original_xaxis, x_arcsec_deflect
        )
        self.interpolate_y_arcsec = RectBivariateSpline(
            self.original_yaxis, self.original_xaxis, y_arcsec_deflect
        )

        self.z_source = z_source
        self.z_lens = z_lens
        self.z_assumed = z_assumed
        if z_source is None:
            self.distance_ratio = 1.0
            self.compute_deflection_angles()
        else:
            assert z_lens is not None
            assert z_assumed is not None
            self.use_redshifts(z_source, z_lens, z_assumed)

    @classmethod
    def create(
        cls,
        *,
        data_or_fname_xy_arcsec_deflect: Optional[
            Union[tuple[np.ndarray, ...], tuple[str, ...]]
        ] = None,
        data_or_fname_xy_pixel_deflect: Optional[
            Union[tuple[np.ndarray, ...], tuple[str, ...]]
        ] = None,
        data_or_fname_psi_arcsec: Optional[Union[np.ndarray, str]] = None,
        header: Optional[fits.Header] = None,
        index_hdul: int = 0,
        z_source: Optional[float] = None,
        z_lens: Optional[float] = None,
        z_assumed: float = np.inf,
    ) -> GravLens:
        '''Constructer of ``GravLens``.

        Either of the first three arguments are required to construct ``GravLens``.
        If more thean one among the three are given, the earier argument take a priority
        (i.e., xy_arcsec_deflect > xy_pixel_deflect > psi_arcsec).

        Args:
            data_or_fname_xy_arcsec_deflect (Optional[Union[tuple[np.ndarray, ...],
                tuple[str, ...]]]): Tuple of the data arrays or fits file names of
                lensing parmeters, x-arcsec-deflect and y-arcsec-deflect.
                Defaults to None.
            data_or_fname_xy_pixel_deflect (Optional[Union[tuple[np.ndarray, ...],
                tuple[str, ...]]]): Tuple of the data arrays or fits file names of
                lensing parmeters, x-pixel-deflect and y-pixel-deflect.
                Defaults to None.
            data_or_fname_psi_arcsec (Optional[Union[np.ndarray, str]]): Data array
                or fits file name of a lensing parmeter, psi. Note that this method
                compute the gradient of psi to obtain the deflection angles, so that
                significantly strong gravitational lensing might not be traced by psi.
                Defaults to None.
            header (Optional[fits.Header], optional): Header of the fits file.
                This method assumes that lensing parameter maps, x-arcsec-deflect and
                y-arcsec-deflect, have the same size and coordinates. Defaults to None.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Defaults to 0.
            z_source (Optional[float], optional): The source (galaxy) redshift.
                Defaults to None.
            z_lens (Optional[float], optional): The lens (cluster) redshift.
                Defaults to None.
            z_assumed (float, optional): The redshift assumed in the gravitational
                parameters. If D_s / D_L = 1, the value should be infinite (``np.inf``).
                Defaults to np.inf.

        Returns:
            GravLens: Instance of ``GravLens``.
        '''
        data_or_fname: Any
        redshifts = (z_source, z_lens, z_assumed)

        if (data_or_fname := data_or_fname_xy_arcsec_deflect) is not None:
            data_or_fname_x, data_or_fname_y = data_or_fname
            if isinstance((x_arcsec := data_or_fname_x), np.ndarray):
                assert isinstance((y_arcsec := data_or_fname_y), np.ndarray)
                return cls(x_arcsec, y_arcsec, header, *redshifts)
            elif isinstance((fname_x := data_or_fname_x), str):
                assert isinstance((fname_y := data_or_fname_y), str)
                loaded = cls.loadfits(fname_x, fname_y, index_hdul=index_hdul)
                return cls(*loaded, *redshifts)

        elif (data_or_fname := data_or_fname_xy_pixel_deflect) is not None:
            data_or_fname_x, data_or_fname_y = data_or_fname
            if isinstance((x_pixel := data_or_fname_x), np.ndarray):
                assert isinstance((y_pixel := data_or_fname_y), np.ndarray)
                x_arcsec, y_arcsec = cls.convert_xy_pixel_to_arcsec(
                    x_pixel, y_pixel, header=header
                )
                return cls(x_arcsec, y_arcsec, header, *redshifts)
            elif isinstance((fname_x := data_or_fname_x), str):
                assert isinstance((fname_y := data_or_fname_y), str)
                loaded = cls.loadfits(fname_x, fname_y, index_hdul=index_hdul)
                x_arcsec, y_arcsec = cls.convert_xy_pixel_to_arcsec(*loaded)
                header = loaded[2]
                return cls(x_arcsec, y_arcsec, header, *redshifts)

        elif (data_or_fname := data_or_fname_psi_arcsec) is not None:
            if isinstance(data := data_or_fname, np.ndarray):
                y_arcsec, x_arcsec = cls.gradient(data, header=header)
                return cls(x_arcsec, y_arcsec, header, *redshifts)
            elif isinstance((fname := data_or_fname), str):
                with fits.open(fname) as hdul:
                    psi = hdul[index_hdul].data
                    header = hdul[index_hdul].header
                y_arcsec, x_arcsec = cls.gradient(psi, header=header)
                return cls(x_arcsec, y_arcsec, header, *redshifts)

        message = (
            'Either "data_or_fname_xy_arcsec_deflect",'
            '"data_or_fname_xy_pixel_deflect", or'
            '"data_or_fname_psi_arcsec" must be input.'
        )
        c.logger.error(message)
        raise TypeError(message)

    def lensing(self, xgrid: np.ndarray, ygrid: np.ndarray) -> tuple[np.ndarray, ...]:
        '''Convert coordinates (x, y) from the image plane to the source plane.

        Args:
            xgrid (np.ndarray): Array of the x coordinates on the image plane.
            ygrid (np.ndarray): Array of the y coordinates on the image plane.

        Returns:
            tuple[np.ndarray, ...]: Tuple of coordinates on the source plane.

        Examples:
            >>> xx_source, yy_source = lensing(xx_image, yy_image)
        '''
        return (xgrid - self.x_pixel_deflect, ygrid - self.y_pixel_deflect)

    def create_interpolate_lensing(
        self, xgrid: np.ndarray, ygrid: np.ndarray
    ) -> LensingInterpolate:
        '''Retrun a LensingInterpolate instance.

        Args:
            xgrid (np.ndarray): 2D array of the x coordinate.
            ygrid (np.ndarray): 2D array of the y coordinate.

        Returns:
            LensingInterpolate: [description]

        Examples:
            >>> lensing_interp = gl.create_interpolate_lensing(xgrid, ygrid)
            >>> x0_s, y0_s = lensing_interp(x0_i, y0_i)
        '''
        xx = np.sort(np.unique(xgrid))
        yy = np.sort(np.unique(ygrid))
        return self.LensingInterpolate(
            xx, yy, self.x_pixel_deflect, self.y_pixel_deflect
        )

    class LensingInterpolate:
        '''Interpolate lensing effects to a data point.

        This class method is implemented to convert the center of the model disk to
        the image plane.
        '''

        def __init__(
            self,
            xx: np.ndarray,
            yy: np.ndarray,
            x_pixel_deflect: np.ndarray,
            y_pixel_deflect: np.ndarray,
        ) -> None:
            self.fx = RectBivariateSpline(yy, xx, x_pixel_deflect)
            self.fy = RectBivariateSpline(yy, xx, y_pixel_deflect)

        def __call__(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
            return np.squeeze(np.array([x - self.fx(y, x), y - self.fy(y, x)]))

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
        wcs_cube = wcs.WCS(cube.header)
        wcs_gl = wcs.WCS(self.header)
        skycoord_wcs, _, _ = wcs_cube.pixel_to_world(
            cube.xgrid[0, :, :].ravel(), cube.ygrid[0, :, :].ravel(), 0, 0
        )
        # self.idx_wcs = wcs_gl.world_to_array_index(skycoord_wcs)
        xpixels, ypixels = wcs_gl.world_to_pixel(skycoord_wcs)
        self.xaxis = np.mean(xpixels.reshape(cube.xgrid.shape[1:]), axis=0)
        self.yaxis = np.mean(ypixels.reshape(cube.ygrid.shape[1:]), axis=1)
        # self.xaxis = np.sort(np.unique(xpixels))
        # self.yaxis = np.sort(np.unique(ypixels))
        # self.shape = cube.xgrid.shape[1:]
        self.header_datacube = cube.header
        self.compute_deflection_angles()

    def match_wcs_with_2d(self, image: np.ndarray, header: fits.Header) -> None:
        '''Match the world coordinate system with an image.

        Utility function to treat lensing maps for 2d images. Match the lens parameter
        coordinates with the image coordinates through wcs.
        Use the image wcs in an input header.

        Args:
            image (np.ndarray): 2d image.
            header (fits.Header): Header of ``image``.

        Returns:
            None:

        Examples:
            >>> gravlens.match_wcs_with(datacube)
        '''
        wcs_image = wcs.WCS(header)
        wcs_gl = wcs.WCS(self.header)

        size_y, size_x = image.shape
        xarray = np.arange(0, size_x)
        yarray = np.arange(0, size_y)
        ygrid, xgrid = np.meshgrid(yarray, xarray, indexing='ij')
        skycoord_wcs = wcs_image.pixel_to_world(xgrid.ravel(), ygrid.ravel())
        xpixels, ypixels = wcs_gl.world_to_pixel(skycoord_wcs)

        self.xaxis = np.mean(xpixels.reshape(xgrid.shape), axis=0)
        self.yaxis = np.mean(ypixels.reshape(ygrid.shape), axis=1)
        # self.xaxis = np.sort(np.unique(xpixels))
        # self.yaxis = np.sort(np.unique(ypixels))
        # self.shape = cube.xgrid.shape[1:]
        self.header_datacube = header
        self.compute_deflection_angles()

    def use_redshifts(
        self, z_source: float, z_lens: float, z_assumed: float = np.inf
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
        self.z_source = z_source
        self.z_lens = z_lens
        self.z_assumed = z_assumed
        self.distance_ratio = self.get_angular_distance_ratio(
            z_source, z_lens, z_assumed
        )
        self.compute_deflection_angles()

    def reset_redshifts(self) -> None:
        '''Reset the redshift infomation.'''
        self.z_lens = None
        self.z_source = None
        self.z_assumed = None
        self.distance_ratio = 1.0
        self.compute_deflection_angles()

    def compute_deflection_angles(self):
        '''Compute deflection angles in arcsec and pixels using redshifts'''
        # x_arcsec_raw = self.original_x_arcsec_deflect[self.idx_wcs].reshape(*self.shape)
        # y_arcsec_raw = self.original_y_arcsec_deflect[self.idx_wcs].reshape(*self.shape)
        x_arcsec_raw = self.interpolate_x_arcsec(self.yaxis, self.xaxis)
        y_arcsec_raw = self.interpolate_y_arcsec(self.yaxis, self.xaxis)
        self.x_arcsec_deflect = x_arcsec_raw * self.distance_ratio
        self.y_arcsec_deflect = y_arcsec_raw * self.distance_ratio
        header = self.header_datacube if self.header_datacube else self.header
        self.x_pixel_deflect, self.y_pixel_deflect = self.convert_xy_arcsec_to_pixel(
            self.x_arcsec_deflect, self.y_arcsec_deflect, header=header
        )

    @staticmethod
    def convert_xy_arcsec_to_pixel(
        x_arcsec: np.ndarray, y_arcsec: np.ndarray, header: fits.Header
    ) -> tuple[np.ndarray, np.ndarray]:
        '''Convert deflection angles of x and y in arcsec to in pixels.

        Use the internal header "CDELT" for the conversion.

        Args:
            x_arcsec (np.ndarray): Deflection angle of x given in arcsec.
            y_arcsec (np.ndarray): Deflection angle of y given in arcsec.
            header (fits.Header): Header including units information

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of the deflection angles of x
                and y in pixels.

        Examples:
            >>> x_pix, y_pix = gl.convert_xy_arcsec_to_pixel(x_arcsec, y_arcsec)

        Nonte:
            Assumes that the units of "CDELT1" and "CDELT2" are degree.
        '''
        dx_arcsec = abs(header['CDELT1'] * 3600)
        dy_arcsec = abs(header['CDELT2'] * 3600)
        return (x_arcsec / dx_arcsec, y_arcsec / dy_arcsec)

    @staticmethod
    def convert_xy_pixel_to_arcsec(
        x_pixel: np.ndarray, y_pixel: np.ndarray, header: fits.Header
    ) -> tuple[np.ndarray, np.ndarray]:
        '''Convert deflection angles of x and y in pixel to in arcsec.

        Use the internal header "CDELT" for the conversion.

        Args:
            x_pixel (np.ndarray): Deflection angle of x given in pixel.
            y_pixel (np.ndarray): Deflection angle of y given in pixel.
            header (fits.Header): Header including units information.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of the deflection angles of x
                and y in arcsec.

        Examples:
            >>> x_arcsec, y_arcsec = gl.convert_xy_pixel_to_arcsec(x_pix, y_pix, header)

        Nonte:
            Assumes that the units of "CDELT1" and "CDELT2" are degree.
        '''
        dx_arcsec = abs(header['CDELT1'] * 3600)
        dy_arcsec = abs(header['CDELT2'] * 3600)
        return (x_pixel * dx_arcsec, y_pixel * dy_arcsec)

    @staticmethod
    def gradient(psi: np.ndarray, header: fits.Header) -> tuple[np.ndarray, np.ndarray]:
        '''Compute gradient of 2D image.

        This method is used to compute deflection angles from the deflection
        potential psi.

        Args:
            psi (np.ndarray): 2D image of the deflection potential. The units are
                given in arcsec, meaning that differentiation of psi with respect
                to the angle in arcsec gives the deflection angles in arcsec.
            header (fits.Header): Header including units information.

        Returns:
            tuple[np.ndarray, np.ndarray]: Tuple of the deflection angles of y
                and x. Note that the order is (y, x).

        Examples:
            >>> y_arcsec, x_arcsec = gl.gradient(psi, header=header)

        Nonte:
            Assumes that the units of "CDELT1" and "CDELT2" are degree.
        '''
        dx_arcsec = abs(header['CDELT1'] * 3600)
        dy_arcsec = abs(header['CDELT2'] * 3600)
        return np.gradient(psi, dy_arcsec, dx_arcsec)

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
            >>> x_deflect_new = x_deflect_old * distance_ratio
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
        fname_x_deflect: str, fname_y_deflect: str, index_hdul: int = 0
    ) -> tuple[np.ndarray, np.ndarray, fits.Header]:
        '''Read gravlens from fits file.

        Args:
            fname_x_deflect (str): Fits file name of the deflect map of x.
            fname_y_deflect (str): Fits file name of the deflect map of y.
            index_hdul (int, optional): Index of fits extensions of the fits file.
                Assumes that all the fits files include the lensing parameter maps
                at the same extension index. Defaults to 0.

        Returns:
            tuple[np.ndarray, np.ndarray, fits.Header]: Tuple of three objects;
                deflection angles (x and y) and header.
        '''
        with fits.open(fname_x_deflect) as hdul:
            x_deflect = hdul[index_hdul].data
            header = hdul[index_hdul].header
        with fits.open(fname_y_deflect) as hdul:
            y_deflect = hdul[index_hdul].data
        return x_deflect, y_deflect, header


class GravLensOld:
    '''Contains gravitational lensing used for Cube.

    Contents are lensing parameters depending on positions: gamma1, gamma2, and kappa.

    Warnig:
        This class is outdated. No longer used.
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
    ) -> GravLensOld:
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
        self, z_source: float, z_lens: float, z_assumed: float = np.inf
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
        '''Reset the redshift infomation.'''
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
        gamma2 = self.gamma1**2 + self.gamma2**2
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
        fname_gamma1: str, fname_gamma2: str, fname_kappa: str, index_hdul: int = 0
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
