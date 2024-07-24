'''User Interface of Tokult.
'''

from __future__ import annotations
from typing import TYPE_CHECKING
from logging import getLogger
from astropy.nddata import NDData

from .core import Core

if TYPE_CHECKING:
    from .fit import Solution, Optimizer
    from .models import AbstractCubeBuilder
    from .mocktelescope import MockTelescope
    from .mockobs import MockObservation

__all__ = ['Tokult']
logger = getLogger(__name__)


##
class Tokult:
    '''User Interface of Tokult.'''

    def __init__(
        self,
        data: NDData | None,
        optimizer: Optimizer | None = None,
        models: AbstractCubeBuilder | None = None,
        telescope: MockTelescope | None = None,
        observation: MockObservation | None = None,
    ) -> None:
        self.core = Core(
            data=data,
            optimizer=optimizer,
            models=models,
            telescope=telescope,
            observation=observation,
        )

    def runfit(self) -> Solution:
        '''Run fitting.'''
        return self.core.runfit()

    @property
    def data(self) -> NDData:
        return self.core.data

    @data.setter
    def data(self, value: NDData) -> None:
        self.core.data = value

    @property
    def optimizer(self) -> Optimizer:
        return self.core.optimizer

    @optimizer.setter
    def optimizer(self, value: Optimizer) -> None:
        self.core.optimizer = value

    @property
    def observation(self) -> MockObservation:
        return self.core.observation

    @observation.setter
    def observation(self, value: MockObservation) -> None:
        self.core.observation = value


# import numpy as np
# from astropy.io import fits
# from multiprocessing.pool import Pool
# from typing import Sequence, Optional, Union, TYPE_CHECKING

# class OldTokult:
#     '''User Interface of Tokult.

#     Users specify data to launch an instance object and start fitting through
#     the instance. This class contains observed data, psf, and lensing parameters,
#     along with useful functions to manipulate data and models.

#     Args:
#         datacube (DataCube):
#         dirtybeam (Optional[DirtyBeam], optional): Defaults to None.
#         gravlens (Optional[GravLens], optional): Defaults to None.

#     Examples:
#         >>> import tokult
#         >>> tok = tokult.Tokult.launch('data.fits', 'psf.fits',
#                                      ('x-arcsec-deflect.fits', 'y-arcsec-deflect.fits'))
#     '''

#     def __init__(
#         self,
#         datacube: DataCube,
#         dirtybeam: Optional[DirtyBeam] = None,
#         gravlens: Optional[GravLens] = None,
#     ) -> None:
#         self.datacube = datacube
#         self.dirtybeam = dirtybeam
#         self.gravlens = gravlens
#         self.modelcube: Optional[ModelCube] = None
#         self.config = c.ConfigParameters()
#         self.core = Core()

#     @classmethod
#     def launch(
#         cls,
#         data: Union[np.ndarray, str],
#         beam: Union[np.ndarray, str, None] = None,
#         gravlens: Union[tuple[np.ndarray, ...], tuple[str, ...], None] = None,
#         header_data: Optional[fits.Header] = None,
#         header_beam: Optional[fits.Header] = None,
#         header_gravlens: Optional[fits.Header] = None,
#         index_data: int = 0,
#         index_beam: int = 0,
#         index_gravlens: int = 0,
#     ) -> Tokult:
#         '''Constructer of ``Tokult``.

#         Construct an instance easier than to use the init constructer.

#         Args:
#             data (Union[np.ndarray, str]): Observed data. The format is a data
#                 array or a fits file name.
#             beam (Union[np.ndarray, str, None], optional): Dirty beam or point
#                 spread function (PSF). The format is a data array or a fits file
#                 name. Defaults to None.
#             gravlens (Union[tuple[np.ndarray, ...], tuple[str, ...]], None, optional):
#                 Gravitational lensing parameters. The format is a tuple containing
#                 the three data array or fits file names of parameters:
#                 x_arcsec_deflect and y_arcsec_deflect. Defaults to None.
#             header_data (Optional[fits.Header], optional): Header of data.
#                 Defaults to None.
#                 This is necessary when the ``data`` is not given in a fits file.
#             header_beam (Optional[fits.Header], optional): Header of psf.
#                 Defaults to None.
#                 This is necessary when ``beam`` is not given in a fits file.
#             header_gravlens (Optional[fits.Header], optional): Header of gravlens.
#                 Defaults to None.
#                 This is necessary when ``gravlens`` is not given in fits files.
#             index_data (int, optional): Index of fits extensions of the data fits
#                 file. Defaults to 0.
#             index_beam (int, optional): Index of fits extensions of the beam fits
#                 file. Defaults to 0.
#             index_gravlens (int, optional): Index of fits extensions of the lens
#                 fits files. Defaults to 0.

#         Examples:
#             >>> tok = tokult.Tokult.launch(
#                          'data.fits', 'psf.fits',
#                          (x_arcsec_deflect.fits', y_arcsec_deflect.fits'))
#         '''
#         datacube = DataCube.create(data, header=header_data, index_hdul=index_data)

#         dirtybeam: Optional[DirtyBeam]
#         if beam is not None:
#             dirtybeam = DirtyBeam.create(
#                 beam, header=header_beam, index_hdul=index_beam
#             )
#         else:
#             dirtybeam = None

#         gl: Optional[GravLens]
#         if gravlens is not None:
#             gl = GravLens.create(
#                 data_or_fname_xy_arcsec_deflect=gravlens,
#                 header=header_gravlens,
#                 index_hdul=index_gravlens,
#             )
#             gl.match_wcs_with(datacube)
#         else:
#             gl = None
#         return cls(datacube, dirtybeam, gl)

#     def imagefit(
#         self,
#         init: Sequence[float],
#         bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
#         fix: Optional[fitting.FixParams] = None,
#         optimization: str = 'mc',
#         niter: int = 1,
#         nperturb: int = 1000,
#         nwalkers: int = 64,
#         nsteps: int = 5000,
#         pool: Optional[Pool] = None,
#         is_separate: bool = False,
#         mask_for_fit: Optional[np.ndarray] = None,
#         uvcoverage: Optional[np.ndarray] = None,
#         progressbar: bool = False,
#     ) -> fitting.Solution:
#         '''Fit a 3d model to the data cube on the image plane.

#         Args:
#             init (Sequence[float]): Initial parameters of fitting.
#             bound (Optional[tuple[Sequence[float], Sequence[float]], optional):
#                 Boundaries of parameters. Defaults to None.
#                 When None is given, the default parameter boundaries are used.
#                 The boundaries can be easily set using ``get_bound_params``.
#                 Currently, in the mcmc method, only flat prior distributions
#                 are available through this argument.
#             fix (Optional[fitting.FixParams], optional): Defaults to None.
#             optimization (str, optional): Defaults to 'mc'.
#             niter (int, optional): Number of iterations of fitting, used
#                 in the least square method. Defaults to 1.
#             nperturb (int, optional): Number of perturbations in the Monte Carlo
#                 method. Defaults to 1000.
#             nwalkers (int, optional): Number of walkers, used in the MCMC method.
#                 Defaults to 64.
#             nsteps (int, optional): Number of steps, used in the MCMC method.
#                 Defaults to 5000.
#             pool (Optional[Pool], optional): multiprocessing.pool for a multi-process
#                 MCMC fitting. Defaults to None.
#             is_separate (bool, optional): If True, parameters regarding kinematics
#                 and emission are separated; and thus fitting uses all the 14
#                 parameters. If False, the parameters are the same and
#                 the number of free parameters are reduced. Defaults to False.
#             mask_for_fit (Optional[np.ndarray], optional): Mask specifying pixels
#                 used for fitting. Defaults to None.
#             progressbar (bool, optional): If True, a progress bar is shown.
#                 Defaults to False.

#         Returns:
#             fitting.Solution: Fitting results and related parameters.

#         Examples:
#             >>> sol = tok.imagefit(init, bound, optimization='mc')
#         '''
#         func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
#         func_lensing = self.gravlens.lensing if self.gravlens else None
#         func_create_lensinginterp = (
#             self.gravlens.create_interpolate_lensing if self.gravlens else None
#         )

#         if optimization == 'mcmc':
#             solution = fitting.mcmc(
#                 self.config,
#                 self.datacube,
#                 init,
#                 bound,
#                 func_convolve=func_convolve,
#                 func_lensing=func_lensing,
#                 func_create_lensinginterp=func_create_lensinginterp,
#                 mode_fit='image',
#                 fix=fix,
#                 is_separate=is_separate,
#                 mask_for_fit=mask_for_fit,
#                 nwalkers=nwalkers,
#                 nsteps=nsteps,
#                 pool=pool,
#                 progressbar=progressbar,
#             )
#         elif optimization == 'ls':
#             solution = fitting.least_square(
#                 self.datacube,
#                 init,
#                 bound,
#                 func_convolve=func_convolve,
#                 func_lensing=func_lensing,
#                 func_create_lensinginterp=func_create_lensinginterp,
#                 niter=niter,
#                 mode_fit='image',
#                 fix=fix,
#                 is_separate=is_separate,
#                 mask_for_fit=mask_for_fit,
#             )
#         elif optimization == 'mc':
#             func_fullconvolve = self.dirtybeam.fullconvolve if self.dirtybeam else None
#             solution = fitting.montecarlo(
#                 self.config,
#                 self.datacube,
#                 init,
#                 bound,
#                 func_convolve=func_convolve,
#                 func_fullconvolve=func_fullconvolve,
#                 func_lensing=func_lensing,
#                 func_create_lensinginterp=func_create_lensinginterp,
#                 niter=niter,
#                 nperturb=nperturb,
#                 fix=fix,
#                 is_separate=is_separate,
#                 mask_for_fit=mask_for_fit,
#                 uvcoverage=uvcoverage,
#                 progressbar=progressbar,
#             )
#         _redshift_tmp = self.gravlens.z_source if self.gravlens is not None else None
#         solution.set_metainfo(z=_redshift_tmp, header=self.datacube.header)
#         self.construct_modelcube(solution.best)
#         return solution

#     def uvfit(
#         self,
#         init: Sequence[float],
#         bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
#         fix: Optional[fitting.FixParams] = None,
#         optimization: str = 'mcmc',
#         niter: int = 1,
#         nwalkers: int = 64,
#         nsteps: int = 5000,
#         pool: Optional[Pool] = None,
#         is_separate: bool = False,
#         mask_for_fit: Optional[np.ndarray] = None,
#         progressbar: bool = False,
#     ) -> fitting.Solution:
#         '''Fit a 3d model to the data cube on the uv plane.

#         Args:
#             init (Sequence[float]): Initial parameters of fitting.
#                 a function to guess initial parameters: ``tokult.initialguess()``.
#             bound (Optional[tuple[Sequence[float], Sequence[float]], optional):
#                 Boundaries of parameters. Defaults to None.
#                 When None is given, the default parameter boundaries are used.
#                 The boundaries can be easily set using ``get_bound_params``.
#                 Currently, in the mcmc method, only flat prior distributions
#                 are available through this argument.
#             fix (Optional[fitting.FixParams], optional): Fix parameters during
#                 fitting. See ``FixParams``. Defaults to None.
#             optimization (str, optional): Method to optimize the 3D model.
#                 - 'ls': least square method.
#                 - 'mcmc': Malcov Chain Monte Carlo method.
#                 Defaults to 'mcmc'.
#             niter (int, optional): Number of iterations of fitting, used
#                 in the least square method. Defaults to 1.
#             nperturb (int, optional): Number of perturbations in the Monte Carlo
#                 method. Defaults to 1000.
#             nwalkers (int, optional): Number of walkers, used in the MCMC method.
#                 Defaults to 64.
#             nsteps (int, optional): Number of steps, used in the MCMC method.
#                 Defaults to 5000.
#             pool (Optional[Pool], optional): multiprocessing.pool for a multi-process
#                 MCMC fitting. Defaults to None.
#             is_separate (bool, optional): If True, parameters regarding kinematics
#                 and emission are separated; and thus fitting uses all the 14
#                 parameters. If False, the parameters are the same and
#                 the number of free parameters are reduced. Defaults to False.
#             mask_for_fit (Optional[np.ndarray], optional): Mask specifying pixels
#                 used for fitting. In uv fitting, specifying the uv-coverage is
#                 important, which is passed through this argument. Defaults to None.
#             progressbar (bool, optional): If True, a progress bar is shown.
#                 Defaults to False.

#         Returns:
#             fitting.Solution: Fitting results and related parameters.

#         Examples:
#             >>> sol = tok.uvfit(init, bound, optimization='mcmc')

#         Note:
#             The input parameters are the same as ``imagefit``.
#         '''
#         if self.dirtybeam:
#             beam_visibility = self.dirtybeam.uvplane
#             norm_weight = self.calculate_normweight()
#         else:
#             msg = '"DirtyBeam" is necessary for uvfit.'
#             logger.error(msg)
#             raise ValueError(msg)
#         func_lensing = self.gravlens.lensing if self.gravlens else None
#         func_create_lensinginterp = (
#             self.gravlens.create_interpolate_lensing if self.gravlens else None
#         )

#         if optimization == 'mcmc':
#             solution = fitting.mcmc(
#                 self.config,
#                 self.datacube,
#                 init,
#                 bound,
#                 beam_vis=beam_visibility,
#                 norm_weight=norm_weight,
#                 func_lensing=func_lensing,
#                 func_create_lensinginterp=func_create_lensinginterp,
#                 mode_fit='uv',
#                 fix=fix,
#                 is_separate=is_separate,
#                 mask_for_fit=mask_for_fit,
#                 nwalkers=nwalkers,
#                 nsteps=nsteps,
#                 pool=pool,
#                 progressbar=progressbar,
#             )
#         elif optimization == 'ls':
#             solution = fitting.least_square(
#                 self.datacube,
#                 init,
#                 bound,
#                 beam_vis=beam_visibility,
#                 norm_weight=norm_weight,
#                 func_lensing=func_lensing,
#                 func_create_lensinginterp=func_create_lensinginterp,
#                 mode_fit='uv',
#                 fix=fix,
#                 is_separate=is_separate,
#                 mask_for_fit=mask_for_fit,
#                 niter=niter,
#             )
#         _redshift_tmp = self.gravlens.z_source if self.gravlens is not None else None
#         solution.set_metainfo(z=_redshift_tmp, header=self.datacube.header)
#         self.construct_modelcube(solution.best)
#         return solution

#     def initialguess(self, is_separate: bool = False) -> fitting.InputParams:
#         '''Guess initial input parameters for fitting.

#         Fit tow-dimensional moment-0 (flux) map and moment-1 (velocity) map then
#         Estimate input parameters from the 2-d fitting results.

#         Args:
#             is_separate (bool, optional): If True, parameters regarding kinematics
#                 and emission are separated; and thus fitting uses all the 14
#                 parameters. If False, the parameters are the same and
#                 the number of free parameters are reduced. Defaults to False.

#         Returns:
#             fitting.InputParams: Best-guessed input parameters.

#         Examples:
#             >>> init = tok.initialguess()

#         Note:
#             Initial parameters are crucial for parameter fitting, especially in
#             the least-square and Monte Carlo methods and being related with speed
#             of convergence in the MCMC method.
#         '''
#         func_convolve = self.dirtybeam.convolve if self.dirtybeam else None
#         func_lensing = self.gravlens.lensing if self.gravlens else None
#         func_create_lensinginterp = (
#             self.gravlens.create_interpolate_lensing if self.gravlens else None
#         )
#         return fitting.initialguess(
#             datacube=self.datacube,
#             func_convolve=func_convolve,
#             func_lensing=func_lensing,
#             func_create_lensinginterp=func_create_lensinginterp,
#             is_separate=is_separate,
#         )

#     def use_region(
#         self,
#         xlim: Optional[tuple[int, int]] = None,
#         ylim: Optional[tuple[int, int]] = None,
#         vlim: Optional[tuple[int, int]] = None,
#     ) -> None:
#         '''Use a region of datacube used for fitting.

#         Args:
#             xlim (Optional[tuple[int, int]], optional): The limit of the x-axis.
#                 Defaults to None.
#             ylim (Optional[tuple[int, int]], optional): The limit of the y-axis.
#                 Defaults to None.
#             vlim (Optional[tuple[int, int]], optional): The limit of the v-axis.
#                 Defaults to None.

#         Returns:
#             None:

#         Examples:
#             >>> tok.use_region((32, 96), (32, 96), (5, 12))

#         Note:
#             In ``uvfit``, the v-axis limit must be specified smaller than
#             original cube size, because ``uvfit`` estimates the noise level
#             using the pixels outside ``vlim``.
#         '''
#         self.datacube.cutout(xlim, ylim, vlim)
#         if self.dirtybeam is not None:
#             self.dirtybeam.cutout_to_match_with(self.datacube)
#         if self.gravlens is not None:
#             self.gravlens.match_wcs_with(self.datacube)

#     def use_redshifts(
#         self, z_source: float, z_lens: float, z_assumed: float = np.inf
#     ) -> None:
#         '''Set the redshifts of the source and the lens galaxies.

#         The redshifts are used to compute the gravitational lensing effects and
#         to convert the parameters to the physical units.

#         Args:
#             z_source (float): The source (galaxy) redshift.
#             z_lens (float): The lens (cluster) redshift.
#             z_assumed (float, optional): The redshift assumed in the
#                 gravitational parameters. If D_s / D_L = 1, the value
#                 should be infinite (``np.inf``). Defaults to ``np.inf``.

#         Returns:
#             None:

#         Examples:
#             >>> tok.use_redshift(z_source=6.2, z_lens=0.9)
#         '''
#         if self.gravlens is not None:
#             self.gravlens.use_redshifts(z_source, z_lens, z_assumed)

#     def change_datacube(
#         self,
#         data_or_fname: Union[np.ndarray, str],
#         header: Optional[fits.Header] = None,
#         index_hdul: int = 0,
#         xlim: Optional[tuple[int, int]] = None,
#         ylim: Optional[tuple[int, int]] = None,
#         vlim: Optional[tuple[int, int]] = None,
#     ) -> None:
#         '''Change the variable ``datacube``.

#         May be useful to change ``datacube`` of an instance.

#         Nonte:
#             This method should be used before ``use_region`` and ``use_redshift``.
#         '''
#         self.datacube = DataCube.create(
#             data_or_fname, header=header, index_hdul=index_hdul
#         )
#         self.use_region(xlim, ylim, vlim)

#     def change_dirtybeam(
#         self,
#         data_or_fname: Union[np.ndarray, str],
#         header: Optional[fits.Header] = None,
#         index_hdul: int = 0,
#     ) -> None:
#         '''Change the variable ``dirtybeam``.

#         May be useful to change ``dirtybeam`` of an instance.

#         Nonte:
#             This method should be used before ``use_region`` and ``use_redshift``.
#         '''
#         self.dirtybeam = DirtyBeam.create(
#             data_or_fname, header=header, index_hdul=index_hdul
#         )

#     def change_gravlens(
#         self,
#         *,
#         data_or_fname_xy_arcsec_deflect: Optional[
#             Union[tuple[np.ndarray, ...], tuple[str, ...]]
#         ] = None,
#         data_or_fname_xy_pixel_deflect: Optional[
#             Union[tuple[np.ndarray, ...], tuple[str, ...]]
#         ] = None,
#         data_or_fname_psi_arcsec: Optional[Union[np.ndarray, str]] = None,
#         header: Optional[fits.Header] = None,
#         index_hdul: int = 0,
#     ) -> None:
#         '''Change the variable ``gravlens``.

#         May be useful to change ``gravlens`` of an instance.

#         Nonte:
#             This method should be used before ``use_region`` and ``use_redshift``.
#         '''
#         self.gravlens = GravLens.create(
#             data_or_fname_xy_arcsec_deflect=data_or_fname_xy_arcsec_deflect,
#             data_or_fname_xy_pixel_deflect=data_or_fname_xy_pixel_deflect,
#             data_or_fname_psi_arcsec=data_or_fname_psi_arcsec,
#             header=header,
#             index_hdul=index_hdul,
#         )
#         self.gravlens.match_wcs_with(self.datacube)

#     def construct_modelcube(self, params: tuple[float, ...]) -> None:
#         '''Construct ``modelcube`` from the input parameters.

#         Construct a 3D model and set it to an internal variable ``modelcube``.
#         If you want to use a model outside ``Tokult`` instances, please use
#         ``ModelCube.create()`` instead.

#         Args:
#             params (tuple[float, ...]): Input parameters.

#         Returns:
#             None:

#         Examples:
#             Change a part of the best-fit parameters and construct the model.

#             >>> params = sol.best._replace(PA_dyn=0.0)
#             >>> tok.construct_modelcube(params)

#         Note:
#             This method needs the global parameters to be set already.
#             You may need to fit the data once, before using this method.
#         '''
#         datacube = self.datacube
#         func_lensing = self.gravlens.lensing if self.gravlens else None
#         func_convolve = self.dirtybeam.fullconvolve if self.dirtybeam else None
#         func_create_lensinginterp = (
#             self.gravlens.create_interpolate_lensing if self.gravlens else None
#         )
#         # fitting.initialize_globalparameters_for_image(
#         #     datacube, func_convolve, func_lensing
#         # )
#         self.modelcube = ModelCube.create(
#             params,
#             datacube=datacube,
#             convolve=func_convolve,
#             lensing=func_lensing,
#             create_interpolate_lensing=func_create_lensinginterp,
#             upsampling_rate=self.config.pixel_upsampling_rate,
#         )

#     def calculate_normweight(self) -> float:
#         '''Calculate the normalization weight used in ``uvfit``.

#         The obtained value is almost equal to sum-of-weight, but different by a
#         factor of a few.

#         Returns:
#             float: The normalization weight
#         '''
#         assert self.dirtybeam is not None
#         uv = self.datacube.rfft2(self.datacube.original)
#         uvpsf = misc.rfft2(self.dirtybeam.original)
#         uvpsf[uvpsf == 0] = misc.min_abs(uvpsf)  # to prevent divide-by-zero
#         uv_noise = uv / np.sqrt(abs(uvpsf.real))

#         # Noise computed from side channels of (v0-1, v1)
#         # Pixels where xlim=(1:-1) should be gaussian noise both in real and imag parts.
#         v0, v1 = self.datacube.vlim
#         n_real = uv_noise[[v0 - 1, v1], :, 1:-1].real
#         n_imag = uv_noise[[v0 - 1, v1], :, 1:-1].imag
#         p = uvpsf[[v0 - 1, v1], :, 1:-1].real
#         n = np.concatenate((n_real[p > -p.min()], n_imag[p > -p.min()]))
#         return 1.0 / n.std() ** 2
