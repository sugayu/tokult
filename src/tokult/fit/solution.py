'''Fitting Solutions
'''

from abc import ABC, abstractmethod
from typing import Optional, Union
from dataclasses import dataclass
import numpy as np
from astropy.io import fits


##
class SolutionDI(ABC):
    '''Absctract class of dependency injector for Solution.'''

    def __init__(self) -> None: ...


class Solution:
    '''Contains output solutions of fittings.'''

    def __init__(
        self,
        p_best: Union[list[float], tuple[float, ...]],
        error_high: Union[list[float], tuple[float, ...]],
        error_low: Union[list[float], tuple[float, ...]],
        chi2: float,
        dof: float,
        cov: np.ndarray,
        mode_fitting: str,
        optimizer: SolutionDI,
        # sampler: Optional[emcee.EnsembleSampler] = None,
        # params_mc: Optional[np.ndarray] = None,
        # output: Optional[OptimizeResult] = None,
    ) -> None:
        # self.best = InputParams(*p_best)
        # self.error_high = InputParams(*error_high)
        # self.error_low = InputParams(*error_low)
        self.chi2 = chi2
        self.dof = dof
        self.cov = cov
        # self.sampler = sampler
        # self.params_mc = params_mc
        # self.output = output
        self.meta = self.MetaInfoOfSolution()

        global parameters_preset, index_free, index_fixp_target, index_fixp_source
        # self.parameters_preset = np.copy(parameters_preset)
        # self.index_free = np.copy(index_free)
        # self.index_fixp_target = np.copy(index_fixp_target)
        # self.index_fixp_source = np.copy(index_fixp_source)

    def set_metainfo(
        self, z: Optional[float] = None, header: Optional[fits.Header] = None
    ) -> None:
        '''Set meta infomation used in Solution'''
        keys_arguments = ['z', 'header']

        for k in keys_arguments:
            if (value_input := locals()[k]) is not None:
                setattr(self.meta, k, value_input)

    @dataclass
    class MetaInfoOfSolution:
        '''Meta data container of Solution.'''

        z: float = 0.0
        header: Optional[fits.Header] = None

    # def add_units(
    #     self, params: Optional[Union[InputParams, np.ndarray]] = None
    # ) -> FitParamsWithUnits:
    #     '''Get best parameters with physical units.'''
    #     if params is None:
    #         return self.best.to_units(header=self.meta.header, redshift=self.meta.z)
    #     elif isinstance(params, InputParams):
    #         return params.to_units(header=self.meta.header, redshift=self.meta.z)
    #     elif isinstance(params, np.ndarray):
    #         return InputParamsArray.from_ndarray(params).to_units(
    #             header=self.meta.header, redshift=self.meta.z
    #         )

    # def restore_params(
    #     self, params: Union[np.ndarray, tuple[float, ...]]
    # ) -> Union[np.ndarray, tuple[float, ...]]:
    #     '''Restore parameters with pfix by inserting parameters into params.'''
    #     if (parameters_preset is None) or (len(params) == 14):
    #         return params

    #     if isinstance(params, np.ndarray):
    #         if params.ndim == 2:
    #             newshape = (params.shape[0], 1)
    #             _parameters_preset = np.tile(self.parameters_preset, newshape).T
    #             _parameters_preset[self.index_free, :] = params
    #             _parameters_preset[self.index_fixp_target, :] = _parameters_preset[
    #                 self.index_fixp_source, :
    #             ]
    #             return _parameters_preset

    #     parameters_preset[self.index_free] = params
    #     parameters_preset[self.index_fixp_target] = parameters_preset[
    #         self.index_fixp_source
    #     ]
    #     return tuple(parameters_preset)
