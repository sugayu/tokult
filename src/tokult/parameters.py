'''Utilities to manipulate parameters.
'''

from __future__ import annotations
from typing import (
    Type,
    NamedTuple,
    dataclass_transform,
    TYPE_CHECKING,
    Any,
    Self,
)
from itertools import accumulate
from collections import namedtuple
from collections.abc import Callable
from dataclasses import dataclass, field, is_dataclass
from copy import deepcopy
import numpy as np
from numpy.random import default_rng
import astropy.units as u

from .utils.dataclass import fields, fieldnames

if TYPE_CHECKING:
    from .mockobs import MockObservation
    from .fit import Optimizer

__all__ = ['FittingParametersBase', 'FitPar', 'ParameterManager']


##
CONTAINER_NAMEDTUPLE: dict[str, Type[NamedTuple]] = {}


@dataclass
class FitPar:
    '''Configuration of fitting parameters.'''

    unit: u.Unit
    bound: tuple[float, float]
    initial: float | None
    fix: float | None = field(init=False, default=None)


@dataclass_transform()
class FittingParametersBase:
    '''Parent class for fitting parameters.

    All the fitting parameter class have to inherit this class.
    '''

    name: str = ''

    def __new__(cls, *args, **kwargs):
        if not is_dataclass(cls):
            for v in vars(cls):
                if not isinstance(p := getattr(cls, v, None), FitPar):
                    continue
                # The deepcopy below is necessary to send a value to dataclass.
                # This may be because dataclass internally deletes default attributes to
                # re-define them in __init__. Subsequently, the defined FitPar instances
                # may be removed from the memory and default_factory returns None.
                setattr(cls, v, field(default_factory=lambda p0=p: deepcopy(p0)))

            dataclass(cls, **kwargs)  # Directly changes cls
        newclass = super().__new__(cls)
        newclass.name = newclass.__class__.__name__
        return newclass

    def namedtuplize(self, values: tuple):
        '''Name elements of a fitting parameter tuple.

        This method is assumed to be used in child classes.

        Args:
            values (tuple): Parameter tuple.

        Examples:
            >>> class NewParameters(FittingParametersBase):
            >>>     x0: FitPar: FitPar(...)
            >>>     y0: FitPar: FitPar(...)
            >>> newp = NewParameters()
            >>> t = newp.namedtuplize((2, 3))
        '''
        clsname = self.__class__.__name__
        if not is_dataclass(self):
            raise TypeError(
                f'namedtuplize of {clsname} cannot be used because it is not dataclass.'
            )

        global CONTAINER_NAMEDTUPLE
        name = clsname + 'Tuple'

        _fields = tuple(field.name for field in fields(self))
        _namtpl = CONTAINER_NAMEDTUPLE.get(name, None)
        if (_namtpl is not None) and (_namtpl._fields == _fields):
            return _namtpl(*values)

        namtpl = namedtuple(name, _fields)  # type: ignore
        CONTAINER_NAMEDTUPLE[name] = namtpl
        return namtpl(*values)


class _DotDict(dict):
    def __getattr__(self, key: str) -> Any:
        return dict.__getitem__(self, key)

    def __setattr__(self, key: str, value: Any) -> None:
        return dict.__setitem__(self, key, value)

    def __delattr__(self, key: str) -> None:
        return dict.__delitem__(self, key)


class ParameterArray(np.ndarray):
    '''Wrapper of numpy.ndarray to show parameters like attributes with dots.'''

    def __new__(cls, array, paramkeys: _DotDict) -> Self:
        obj = np.asarray(array).view(cls)
        _nkeys1 = list(accumulate([len(k) for k in paramkeys.values()]))
        _nkeys0 = [0] + _nkeys1[:-1]
        _nkeys = [slice(n0, n1) for n0, n1 in zip(_nkeys0, _nkeys1)]
        _keyslice = _DotDict(zip(paramkeys.keys(), _nkeys))
        setattr(obj, '_keys', paramkeys)
        setattr(obj, '_keyslice', _keyslice)
        return obj

    def __array_finalize__(self, obj) -> None:
        if obj is None:
            return
        setattr(self, '_keys', getattr(obj, '_keys', None))
        setattr(self, '_keyslice', getattr(obj, '_keyslice', None))

    def __getitem__(self, key):
        if isinstance(key, str):
            return _DotDict(zip(self._keys[key], self[self._keyslice[key]]))
        else:
            return super().__getitem__(key)

    __getattr__ = __getitem__


class ParameterManager:
    '''Summary of all the fitting parameters.

    This class controles behaviour of all the fitting parameters for model building,
    mock observations, and fitting formulae.
    '''

    def __init__(self, mockobs: MockObservation, optimizer: Optimizer) -> None:
        # Attributes
        # As dict holds the order from Python 3.7, dict is used instead of OrderedDict.
        self.parameters: dict[str, FittingParametersBase] = dict()
        self.nmax: int
        # self.nparams: list[int] = []
        self._slices: list[slice] = []

        self._index_free: np.ndarray = np.array([])

        self._index_fix_float: np.ndarray
        self._fixed_values: np.ndarray

        self._index_fixp_from: np.ndarray
        self._index_fixp_to: np.ndarray

        self._index_func: np.ndarray
        self._list_func: list[Callable]

        self._paramkeys: _DotDict
        self._paramindices: dict[str, dict[str, int]] = {}
        self._initialvalues: np.ndarray
        self.bounds: np.ndarray

        # Initialize
        self.register(mockobs.models.galaxies.kinematic_model)
        self.register(mockobs.models.galaxies.brightness_model)
        self.register(mockobs.models.galaxies)
        self.register(mockobs.telescope.components)
        self.register(optimizer)

        self.standby()

    def standby(self) -> None:
        '''Prepare attibutes and methods to manipulate parameters.'''
        # Set paramkeys
        self._paramkeys = _DotDict(
            [(key, fieldnames(p)) for key, p in self.parameters.items()]
        )

        # Set slices and paramindices
        # _nkeys1 = list(accumulate(self.nparams))
        _nkeys1 = list(
            accumulate([len(_fields) for _fields in self._paramkeys.values()])
        )
        _nkeys0 = [0] + _nkeys1[:-1]
        self._slices = [slice(n0, n1) for n0, n1 in zip(_nkeys0, _nkeys1)]
        for (k, fp), n0 in zip(self._paramkeys.items(), _nkeys0):
            self._paramindices[k] = {p: n0 + i for i, p in enumerate(fp)}
            self.nmax = max(self._paramindices[k].values()) + 1

        # Set attributes for .restore()
        index_free = np.zeros(self.nmax).astype(bool)
        index_fix_float = np.zeros(self.nmax).astype(bool)
        fixed_values = []
        index_fixp_from: list[int] = []
        index_fixp_to = np.zeros(self.nmax).astype(bool)
        index_func = np.zeros(self.nmax).astype(bool)
        list_func: list[Callable] = []
        initialvalues: list[float] = []
        bounds: list[tuple[float, float]] = []
        for name, parambase in self.parameters.items():
            for pname in fieldnames(parambase):

                p: FitPar = getattr(parambase, pname)
                i = self._paramindices[name][pname]

                if p.initial is not None:
                    initialvalues.append(p.initial)
                else:
                    initialvalues.append((p.bound[0] + p.bound[1]) / 2.0)

                bounds.append(p.bound)

                if p.fix is None:
                    index_free[i] = True
                if isinstance(p.fix, float):
                    index_fix_float[i] = True
                    fixed_values.append(p.fix)
                if isinstance(p.fix, str):
                    key0, key1 = p.fix.split('.')
                    index_fixp_from.append(self._paramindices[key0][key1])
                    index_fixp_to[i] = True
                if callable(p.fix):
                    index_func[i] = True
                    list_func.append(p.fix)

        self._index_free = np.asarray(index_free)
        self._index_fix_float = np.asarray(index_fix_float)
        self._fixed_values = np.asarray(fixed_values)
        self._index_fixp_from = np.asarray(index_fixp_from)
        self._index_fixp_to = np.asarray(index_fixp_to)
        self._index_func = np.asarray(index_func)
        self._list_func = list_func
        self._initialvalues = np.array(initialvalues)
        self.bounds = np.array(bounds).T[:, self._index_free]

    def register(self, klass: object | list[object] | list[object | None]) -> None:
        '''Add fitting parameters to internal dict to make the complete fit-par list.'''
        if isinstance(klass, list):
            for kls in klass:
                if getattr(kls, 'p', None) is None:
                    continue
                self._register(kls)
        else:
            if getattr(klass, 'p', None) is None:
                return
            self._register(klass)

    def _register(self, klass: object) -> None:
        '''Work for method "register"'''
        p = getattr(klass, 'p', None)
        assert p is not None
        if not isinstance(p, FittingParametersBase):
            raise TypeError(
                f'Input class {p.__class__.__name__} is not FittingParametersBase.'
            )
        name = p.name
        self.parameters[name] = p
        # self.nparams.append(len(fields(p)))

    def initialvalues(
        self, initial: np.ndarray | None = None, seed: int | None = None, ndim: int = 1
    ) -> np.ndarray:
        init = self._initialvalues[self._index_free] if initial is None else initial
        if ndim != 1:
            init = np.tile(init, (ndim, 1))
        if seed is not None:
            rng = default_rng(seed)
            fluctuation = 1e-3 * rng.standard_normal(init.shape)
            init += init * fluctuation
            init[init == 0.0] += fluctuation[init == 0.0]
        return init

    def restore(self, short_parameters: tuple[float, ...]) -> tuple[float, ...]:
        '''Restore a parameter tuple with the complete length.

        The paraemter tuple is shortened in the fitting procedure, because some of the
        paremeters are fixed or determined by other fitting parameters. This class
        restore the complete parmeter tuple by compensating the fixed parameters.

        Args:
            p (tuple[float): Shortened paramters.

        Returns:
            tuple[float]: Complete parameters.
        '''
        if len(short_parameters) != self.nparams:
            raise ValueError(
                f'The length of the input parameters {len(short_parameters)} is '
                f'different from the expected length {self.nparams}.'
            )
        empty_array = np.full_like(self._index_free, np.nan, dtype=float)
        # TODO: This initialization of _fullparam might be skipped from the 2nd cycle.
        fullparams = ParameterArray(empty_array, paramkeys=self._paramkeys)

        # where are free parameters
        fullparams[self._index_free] = short_parameters

        # where are fixed values
        if np.any(self._index_fix_float):
            fullparams[self._index_fix_float] = self._fixed_values

        # where are tighted to other parameters
        if np.any(self._index_fixp_to):
            fullparams[self._index_fixp_to] = fullparams[self._index_fixp_from]

        # where are computed in functions
        if np.any(self._index_func):
            fullparams[self._index_func] = [f(fullparams) for f in self._list_func]

        if np.any(np.isnan(fullparams)):
            raise ValueError(
                'Some of the fitting parameters are not well-defined, including None: '
                f'{fullparams}'
            )

        return tuple(fullparams)

    def shorten(self, full_parameters: tuple[float, ...]) -> tuple[float, ...]:
        '''Shorten the full parameters to the "net" fitting parameters.

        Here the "net" fitting parameters means parameters that are extracted from the
        full_parameters by removing fixed or tighted parameters.

        Args:
            full_parameters (tuple[float, ...]): Parameter tuple, which has the same
                length as all the parameters.

        Returns:
            tuple[float, ...]: Shortened (net) fitting parameters.
        '''
        assert len(full_parameters) == self.nmax
        return tuple(np.array(full_parameters)[self._index_free])

    def extract(
        self, full_parameters: tuple[float, ...], name: str
    ) -> tuple[float, ...]:
        '''Extract parameters belonging to a specified model.

        Args:
            full_parameters (tuple[float, ...]): Parameter tuple, which has the same
                length as all the parameters.
            name (str): Model name.

        Returns:
            tuple[float, ...]: Parameter tuple for the named model.
        '''
        index = list(self.parameters.keys()).index(name)
        return full_parameters[self._slices[index]]

    @property
    def nparams(self) -> int:
        return np.count_nonzero(self._index_free)


# @dataclass
# class FitParamsWithUnits:
#     '''Fitting parameters with units.'''

#     x0_dyn: u.Quantity
#     y0_dyn: u.Quantity
#     PA_dyn: u.Quantity
#     inclination_dyn: u.Quantity
#     radius_dyn: u.Quantity
#     velocity_sys: u.Quantity
#     mass_dyn: u.Quantity
#     brightness_center: u.Quantity
#     velocity_dispersion: u.Quantity
#     radius_emi: u.Quantity
#     x0_emi: u.Quantity
#     y0_emi: u.Quantity
#     PA_emi: u.Quantity
#     inclination_emi: u.Quantity
#     header: Optional[fits.Header] = field(default=None, repr=False)
#     z: float = field(default=0.0, repr=False)
#     wcs: Optional[WCS] = field(init=False, repr=False)
#     pixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
#     freq_rest: Optional[u.Quantity] = field(init=False, repr=False)
#     vpixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
#     diskmassscale: Optional[u.Equivalency] = field(init=False, repr=False)

#     def __post_init__(self) -> None:
#         if self.header:
#             self.wcs = WCS(self.header)
#             self.freq_rest = self.header['RESTFRQ'] * u.Hz

#             deg_pix = abs(self.header['CDELT1']) * u.Unit(self.header['CUNIT1']) / u.pix
#             self.pixelscale = misc.pixel_scale(deg_pix.to(u.arcsec / u.pix), self.z)

#             dfreq_pix = abs(self.header['CDELT3']) * u.Unit(self.header['CUNIT3'])
#             opt_equiv = u.doppler_optical(self.freq_rest)
#             dv_pix = (self.freq_rest - dfreq_pix).to(u.km / u.s, opt_equiv)
#             self.vpixelscale = misc.vpixel_scale(dv_pix / u.pix)

#             self.diskmassscale = (
#                 misc.diskmass_scale(self.pixelscale, self.vpixelscale)
#                 if self.z > 0.0
#                 else None
#             )

#         else:
#             self.wcs = None
#             self.pixelscale = None
#             self.freq_rest = None
#             self.vpixelscale = None
#             self.diskmassscale = None

#     def to_physicalscale(self) -> None:
#         '''Convert values to physicalscales.'''
#         if self.header is None:
#             raise ValueError('header is not input.')
#         assert isinstance(self.wcs, WCS)

#         wcs_celestial = self.wcs.celestial
#         wcs_spectral = self.wcs.spectral
#         coord_celestial = wcs_celestial.pixel_to_world(
#             [self.x0_dyn, self.x0_emi], [self.y0_dyn, self.y0_emi]
#         )
#         coord_spectral = wcs_spectral.pixel_to_world(self.velocity_sys)
#         coord_spectral_kms = coord_spectral.to(
#             u.km / u.s, doppler_convention='optical', doppler_rest=self.freq_rest
#         )
#         self.x0_dyn = coord_celestial.ra[0]
#         self.y0_dyn = coord_celestial.dec[0]
#         self.x0_emi = coord_celestial.ra[1]
#         self.y0_emi = coord_celestial.dec[1]
#         self.velocity_sys = coord_spectral_kms.quantity

#         self.radius_dyn = self.radius_dyn.to(u.arcsec, self.pixelscale)
#         self.radius_emi = self.radius_emi.to(u.arcsec, self.pixelscale)
#         self.brightness_center = self.brightness_center.to(
#             u.Jy / u.arcsec**2, self.pixelscale
#         )
#         self.velocity_dispersion = self.velocity_dispersion.to(
#             u.km / u.s, self.vpixelscale
#         )

#         if self.z > 0.0:
#             self.radius_dyn = self.radius_dyn.to(u.kpc, self.pixelscale)
#             self.radius_emi = self.radius_emi.to(u.kpc, self.pixelscale)
#             self.mass_dyn = self.mass_dyn.physical.to(u.Msun, self.diskmassscale)

#     def vmax(self):
#         '''Maximum rotation velcoity.'''
#         return func.maximum_rotation_velocity(self.mass_dyn, self.radius_dyn)

#     @classmethod
#     def from_inputparams(
#         cls,
#         inputparams: InputParams | InputParamsArray,
#         header: fits.Header | None = None,
#         z: float = 0.0,
#     ) -> FitParamsWithUnits:
#         '''Constructer from InputParams'''
#         dictionary = inputparams._asdict()
#         input_dict = {}
#         units = (
#             u.dimensionless_unscaled,
#             u.dimensionless_unscaled,
#             u.rad,
#             u.rad,
#             u.pix,
#             u.pix,
#             u.dex(u.pix**3),
#             u.Jy / u.pix / u.pix,
#             u.pix,
#             u.pix,
#             u.dimensionless_unscaled,
#             u.dimensionless_unscaled,
#             u.rad,
#             u.rad,
#         )
#         for (key, value), unit in zip(dictionary.items(), units):
#             input_dict[key] = value * unit

#         if header is None:
#             return cls(**input_dict)
#         else:
#             clsself = cls(header=header, z=z, **input_dict)
#             clsself.to_physicalscale()
#             return clsself


# class InputParams(NamedTuple):
#     '''Input parameters for construct_model_at_imageplane.'''

#     x0_dyn: float  #: the coordinate on x-axis
#     y0_dyn: float
#     PA_dyn: float
#     inclination_dyn: float
#     radius_dyn: float
#     velocity_sys: float
#     mass_dyn: float
#     brightness_center: float
#     velocity_dispersion: float
#     radius_emi: float
#     x0_emi: float
#     y0_emi: float
#     PA_emi: float
#     inclination_emi: float

#     def to_units(
#         self, header: fits.Header, redshift: float = 0.0
#     ) -> FitParamsWithUnits:
#         '''Return input parameters with units.'''
#         return FitParamsWithUnits.from_inputparams(self, header, redshift)

# def is_init_outside_of_bound(
#     init: tuple[float, ...], bound: tuple[tuple[float, ...], tuple[float, ...]]
# ) -> bool:
#     '''Return True if init is outside of bound.'''
#     bound0, bound1 = bound
#     for i, b0, b1 in zip(init, bound0, bound1):
#         if (i < b0) or (b1 < i):
#             return True
#     return False


# def shorten_init_and_bound_ifneeded(
#     init: Sequence[float], bound: tuple[Sequence[float], Sequence[float]]
# ) -> tuple[tuple[float, ...], tuple[tuple[float, ...], tuple[float, ...]]]:
#     '''Shorten init and bound parameter to match appropreate lengths.'''
#     global index_free
#     new_init = tuple(np.array(init)[index_free])
#     new_bound0 = tuple(np.array(bound[0])[index_free])
#     new_bound1 = tuple(np.array(bound[1])[index_free])
#     return new_init, (new_bound0, new_bound1)
