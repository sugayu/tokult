'''Least square fitting.
'''

from scipy.optimize import least_squares as sp_least_squares
from scipy.optimize import OptimizeResult
from ..optimize import Optimizer
from ..solution import Solution


__all__ = ['LeastSquare']


##
class LeastSquare(Optimizer):
    ''' '''

    def __init__(self) -> None: ...


#     def least_square(
#         datacube: DataCube,
#         init: Sequence[float],
#         bound: Optional[tuple[Sequence[float], Sequence[float]]] = None,
#         fix: Optional[FixParams] = None,
#         func_convolve: Optional[Callable] = None,
#         func_lensing: Optional[Callable] = None,
#         func_create_lensinginterp: Optional[Callable] = None,
#         beam_vis: Optional[np.ndarray] = None,
#         norm_weight: Optional[float] = None,
#         mask_for_fit: Optional[np.ndarray] = None,
#         niter: int = 1,
#         mode_fit: str = 'image',
#         is_separate: bool = False,
#     ) -> Solution:
#         '''Least square fitting using scipy.optimize.least_squares'''
#         if mask_for_fit is None:
#             mask_for_fit = np.ones_like(datacube.imageplane).astype(bool)
#         if mode_fit == 'image':
#             initialize_globalparameters_for_image(
#                 datacube,
#                 mask_for_fit,
#                 func_convolve,
#                 func_lensing,
#                 func_create_lensinginterp,
#             )
#             func_fit = construct_convolvedmodel
#         elif mode_fit == 'uv':
#             if beam_vis is None:
#                 raise ValueError('"beam_vis" is necessary for uvfit')
#             if norm_weight is None:
#                 raise ValueError('Param "norm_weight" is necessary for uvfit.')
#             if mask_for_fit is None:
#                 mask_for_fit = np.ones_like(datacube.uvplane).astype(bool)

#             initialize_globalparameters_for_uv(
#                 datacube,
#                 beam_vis,
#                 norm_weight,
#                 mask_for_fit,
#                 func_lensing,
#                 func_create_lensinginterp,
#             )
#             func_fit = construct_uvmodel
#         else:
#             raise ValueError(
#                 f'mode_fit is "image" or "uv", no option for "{mode_fit}".'
#             )

#         set_fixedparameters(fix, is_separate)
#         bound = get_bound_params() if bound is None else bound
#         _init, _bound = shorten_init_and_bound_ifneeded(init, bound)
#         if is_init_outside_of_bound(_init, _bound):
#             raise ValueError('The "init" is outside of the "bound".')
#         args = (func_fit,)

#         for _ in range(niter):
#             output = sp_least_squares(calculate_chi, _init, args=args, bounds=_bound)
#             _init = output.x

#         dof = datacube.imageplane.size - 1 - len(_init)
#         chi2 = np.sum(calculate_chi(output.x, func_fit) ** 2.0)
#         return Solution.from_leastsquare(output, chi2, dof)


# class LeastSquareDI(SolutionDI):
#     ''' '''

#     def __init__(self) -> None: ...

#     def from_leastsquare(
#         cls, output: OptimizeResult, chi2: float, dof: float
#     ) -> Solution:
#         '''Construct Solution() from output of least_square.'''
#         p_bestfit = output.x
#         J = output.jac
#         # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
#         cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
#         result_error = np.sqrt(np.diag(cov))
#         p_bestfit = restore_params(p_bestfit)
#         result_error = restore_params(result_error)
#         return Solution(
#             p_bestfit,
#             result_error,
#             result_error,
#             chi2,
#             dof,
#             cov,
#             mode_fitting='leastsquare',
#         )
