'''Utilities to manipulate parameters.
'''

from typing import Type, NamedTuple, dataclass_transform
from collections import namedtuple
from dataclasses import dataclass, field, fields, is_dataclass
import astropy.units as u

__all__ = ['FittingParametersBase', 'FitPar']


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

    def __new__(cls, *args, **kwargs):
        if not is_dataclass(cls):
            dataclass(cls)  # Directly changes cls
        return super().__new__(cls)

    def namedtuplize(self, values: tuple):
        '''Name elements of a fitting parameter tuple.

        This method is assumed to be used in child classes.

        Args:
            values (tuple): Parameter tuple.

        Examples:
            >>> class NewParameters(FittingParametersBase):
            >>>     x0: Parameter(0)
            >>>     y0: Parameter(1)
            >>> newp = NewParameters()
            >>> t = newp.namedtuplize((2, 3))
        '''
        clsname = self.__class__.__name__
        if not is_dataclass(self):
            raise TypeError(f'namedtuplize of {clsname} cannot be used.')

        global CONTAINER_NAMEDTUPLE
        name = clsname + 'Tuple'

        _fields = tuple(field.name for field in fields(self))
        _namtpl = CONTAINER_NAMEDTUPLE.get(name, None)
        if (_namtpl is not None) and (_namtpl._fields == _fields):
            return _namtpl(*values)

        namtpl = namedtuple(name, _fields)  # type: ignore
        CONTAINER_NAMEDTUPLE[name] = namtpl
        return namtpl(*values)


class CompleteFittingParameters:
    '''Summary of all the fitting parameters.

    This class controles behaviour of all the fitting parameters for model building,
    mock observations, and fitting formulae.
    '''

    def __init__(self) -> None:
        self.parameters = None

    def ready(self) -> tuple[float, ...]:
        self.fixes = [p.fix for p in self.p]

    def restore_params(self, p: tuple[float, ...]) -> tuple[float]:
        '''Restore a parameter tuple with the complete length.

        The paraemter tuple is shortened in the fitting procedure, because some of the
        paremeters are fixed or determined by other fitting parameters. This class
        restore the complete parmeter tuple by compensating the fixed parameters.

        Args:
            p (tuple[float): Shortened paramters.

        Returns:
            tuple[float]: Complete parameters.
        '''
        # XXX: TBD
        global parameters_preset, index_free, index_fixp_target, index_fixp_source
        if (parameters_preset is None) or (len(p) == 14):
            return p
        parameters_preset[index_free] = p
        parameters_preset[index_fixp_target] = parameters_preset[index_fixp_source]
        return list(parameters_preset)

    def pop_param(self, p: list[float]) -> tuple[float, ...]:
        '''Pop out a decided length of params from the beggining.

        Args:
            p (list[float]): fitting parameters.

        Returns:
            tuple[float, ...]: parameters used in the brightness model.

        Note:
            This method shorten (change) the input "p".
        '''
        if not isinstance(p, list):
            raise Warning(
                f'The input p must be list, but it has a different type of {type(p)}.'
            )
        _output = p[: self.psize]
        del p[: self.psize]
        return tuple(_output)

    def set_fitpars(self, p: FittingParametersBase) -> None:
        '''Set fitting parameters.'''
        self.p: FittingParametersBase = p
        self.psize = len(fields(self.p))


@dataclass
class FitParamsWithUnits:
    '''Fitting parameters with units.'''

    x0_dyn: u.Quantity
    y0_dyn: u.Quantity
    PA_dyn: u.Quantity
    inclination_dyn: u.Quantity
    radius_dyn: u.Quantity
    velocity_sys: u.Quantity
    mass_dyn: u.Quantity
    brightness_center: u.Quantity
    velocity_dispersion: u.Quantity
    radius_emi: u.Quantity
    x0_emi: u.Quantity
    y0_emi: u.Quantity
    PA_emi: u.Quantity
    inclination_emi: u.Quantity
    header: Optional[fits.Header] = field(default=None, repr=False)
    z: float = field(default=0.0, repr=False)
    wcs: Optional[WCS] = field(init=False, repr=False)
    pixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
    freq_rest: Optional[u.Quantity] = field(init=False, repr=False)
    vpixelscale: Optional[u.Equivalency] = field(init=False, repr=False)
    diskmassscale: Optional[u.Equivalency] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.header:
            self.wcs = WCS(self.header)
            self.freq_rest = self.header['RESTFRQ'] * u.Hz

            deg_pix = abs(self.header['CDELT1']) * u.Unit(self.header['CUNIT1']) / u.pix
            self.pixelscale = misc.pixel_scale(deg_pix.to(u.arcsec / u.pix), self.z)

            dfreq_pix = abs(self.header['CDELT3']) * u.Unit(self.header['CUNIT3'])
            opt_equiv = u.doppler_optical(self.freq_rest)
            dv_pix = (self.freq_rest - dfreq_pix).to(u.km / u.s, opt_equiv)
            self.vpixelscale = misc.vpixel_scale(dv_pix / u.pix)

            self.diskmassscale = (
                misc.diskmass_scale(self.pixelscale, self.vpixelscale)
                if self.z > 0.0
                else None
            )

        else:
            self.wcs = None
            self.pixelscale = None
            self.freq_rest = None
            self.vpixelscale = None
            self.diskmassscale = None

    def to_physicalscale(self) -> None:
        '''Convert values to physicalscales.'''
        if self.header is None:
            raise ValueError('header is not input.')
        assert isinstance(self.wcs, WCS)

        wcs_celestial = self.wcs.celestial
        wcs_spectral = self.wcs.spectral
        coord_celestial = wcs_celestial.pixel_to_world(
            [self.x0_dyn, self.x0_emi], [self.y0_dyn, self.y0_emi]
        )
        coord_spectral = wcs_spectral.pixel_to_world(self.velocity_sys)
        coord_spectral_kms = coord_spectral.to(
            u.km / u.s, doppler_convention='optical', doppler_rest=self.freq_rest
        )
        self.x0_dyn = coord_celestial.ra[0]
        self.y0_dyn = coord_celestial.dec[0]
        self.x0_emi = coord_celestial.ra[1]
        self.y0_emi = coord_celestial.dec[1]
        self.velocity_sys = coord_spectral_kms.quantity

        self.radius_dyn = self.radius_dyn.to(u.arcsec, self.pixelscale)
        self.radius_emi = self.radius_emi.to(u.arcsec, self.pixelscale)
        self.brightness_center = self.brightness_center.to(
            u.Jy / u.arcsec**2, self.pixelscale
        )
        self.velocity_dispersion = self.velocity_dispersion.to(
            u.km / u.s, self.vpixelscale
        )

        if self.z > 0.0:
            self.radius_dyn = self.radius_dyn.to(u.kpc, self.pixelscale)
            self.radius_emi = self.radius_emi.to(u.kpc, self.pixelscale)
            self.mass_dyn = self.mass_dyn.physical.to(u.Msun, self.diskmassscale)

    def vmax(self):
        '''Maximum rotation velcoity.'''
        return func.maximum_rotation_velocity(self.mass_dyn, self.radius_dyn)

    @classmethod
    def from_inputparams(
        cls,
        inputparams: InputParams | InputParamsArray,
        header: fits.Header | None = None,
        z: float = 0.0,
    ) -> FitParamsWithUnits:
        '''Constructer from InputParams'''
        dictionary = inputparams._asdict()
        input_dict = {}
        units = (
            u.dimensionless_unscaled,
            u.dimensionless_unscaled,
            u.rad,
            u.rad,
            u.pix,
            u.pix,
            u.dex(u.pix**3),
            u.Jy / u.pix / u.pix,
            u.pix,
            u.pix,
            u.dimensionless_unscaled,
            u.dimensionless_unscaled,
            u.rad,
            u.rad,
        )
        for (key, value), unit in zip(dictionary.items(), units):
            input_dict[key] = value * unit

        if header is None:
            return cls(**input_dict)
        else:
            clsself = cls(header=header, z=z, **input_dict)
            clsself.to_physicalscale()
            return clsself


class InputParams(NamedTuple):
    '''Input parameters for construct_model_at_imageplane.'''

    x0_dyn: float  #: the coordinate on x-axis
    y0_dyn: float
    PA_dyn: float
    inclination_dyn: float
    radius_dyn: float
    velocity_sys: float
    mass_dyn: float
    brightness_center: float
    velocity_dispersion: float
    radius_emi: float
    x0_emi: float
    y0_emi: float
    PA_emi: float
    inclination_emi: float

    def to_units(
        self, header: fits.Header, redshift: float = 0.0
    ) -> FitParamsWithUnits:
        '''Return input parameters with units.'''
        return FitParamsWithUnits.from_inputparams(self, header, redshift)


class InputParamsArray(NamedTuple):
    '''Input parameter array for construct_model_at_imageplane.'''

    x0_dyn: np.ndarray
    y0_dyn: np.ndarray
    PA_dyn: np.ndarray
    inclination_dyn: np.ndarray
    radius_dyn: np.ndarray
    velocity_sys: np.ndarray
    mass_dyn: np.ndarray
    brightness_center: np.ndarray
    velocity_dispersion: np.ndarray
    radius_emi: np.ndarray
    x0_emi: np.ndarray
    y0_emi: np.ndarray
    PA_emi: np.ndarray
    inclination_emi: np.ndarray

    def to_units(
        self, header: fits.Header, redshift: float = 0.0
    ) -> FitParamsWithUnits:
        '''Return input parameters with units.'''
        return FitParamsWithUnits.from_inputparams(self, header, redshift)

    @classmethod
    def from_ndarray(cls, params: np.ndarray):
        pass


def get_bound_params(
    x0_dyn: tuple[float, float] = (-np.inf, np.inf),
    y0_dyn: tuple[float, float] = (-np.inf, np.inf),
    PA_dyn: tuple[float, float] = (0.0, 2 * np.pi),
    inclination_dyn: tuple[float, float] = (0.0, np.pi / 2),
    radius_dyn: tuple[float, float] = (0.0, np.inf),
    velocity_sys: tuple[float, float] = (-np.inf, np.inf),
    mass_dyn: tuple[float, float] = (-np.inf, np.inf),
    brightness_center: tuple[float, float] = (0.0, np.inf),
    velocity_dispersion: tuple[float, float] = (0.0, np.inf),
    radius_emi: tuple[float, float] = (0.0, np.inf),
    x0_emi: tuple[float, float] = (-np.inf, np.inf),
    y0_emi: tuple[float, float] = (-np.inf, np.inf),
    PA_emi: tuple[float, float] = (0.0, 2 * np.pi),
    inclination_emi: tuple[float, float] = (0.0, np.pi / 2),
) -> tuple[InputParams, InputParams]:
    '''Return bound parameters.'''

    def _bound(i: int) -> InputParams:
        return InputParams(
            x0_dyn=x0_dyn[i],
            y0_dyn=y0_dyn[i],
            PA_dyn=PA_dyn[i],
            inclination_dyn=inclination_dyn[i],
            radius_dyn=radius_dyn[i],
            velocity_sys=velocity_sys[i],
            mass_dyn=mass_dyn[i],
            brightness_center=brightness_center[i],
            velocity_dispersion=velocity_dispersion[i],
            radius_emi=radius_emi[i],
            x0_emi=x0_emi[i],
            y0_emi=y0_emi[i],
            PA_emi=PA_emi[i],
            inclination_emi=inclination_emi[i],
        )

    lower, upper = (0, 1)
    return (_bound(lower), _bound(upper))


def is_init_outside_of_bound(
    init: tuple[float, ...], bound: tuple[tuple[float, ...], tuple[float, ...]]
) -> bool:
    '''Return True if init is outside of bound.'''
    bound0, bound1 = bound
    for i, b0, b1 in zip(init, bound0, bound1):
        if (i < b0) or (b1 < i):
            return True
    return False


class FixParams(NamedTuple):
    '''Fixed parameters for construct_model_at_imageplane.'''

    x0_dyn: Optional[float] = None
    y0_dyn: Optional[float] = None
    PA_dyn: Optional[float] = None
    inclination_dyn: Optional[float] = None
    radius_dyn: Optional[float] = None
    velocity_sys: Optional[float] = None
    mass_dyn: Optional[float] = None
    brightness_center: Optional[float] = None
    velocity_dispersion: Optional[float] = None
    radius_emi: Optional[Union[float, bool]] = None
    x0_emi: Optional[Union[float, bool]] = None
    y0_emi: Optional[Union[float, bool]] = None
    PA_emi: Optional[Union[float, bool]] = None
    inclination_emi: Optional[Union[float, bool]] = None


def set_fixedparameters(fix: Optional[FixParams], is_separate: bool) -> None:
    '''Set global parameters related with fixed parameters.

    if a value in fix is:
    - None: the parameter is not fixed
    - float: the perameter is fixed to the float value
    - True: the parameter has the same value as another parameter
    '''
    global parameters_preset, index_free, index_fixp_target, index_fixp_source
    parameters_preset = np.empty(14)
    index_free = []
    index_fixp_target = []
    index_fixp_source = []

    parameters_fixp = FixParams(
        radius_emi=4, x0_emi=0, y0_emi=1, PA_emi=2, inclination_emi=3
    )
    free_parameter = None
    fixed_to_another_parameter = True

    if (fix is None) and (is_separate):
        parameters_preset = None
        return
    elif fix is None:
        _fix = FixParams()
    else:
        _fix = fix

    if is_separate is False:
        _fix = _fix._replace(
            radius_emi=True, x0_emi=True, y0_emi=True, PA_emi=True, inclination_emi=True
        )

    for i, p in enumerate(_fix):
        p_is_fixed_to_a_value = isinstance(p, float)

        if p_is_fixed_to_a_value:
            parameters_preset[i] = p

        if p is free_parameter:
            index_free.append(i)

        if p is fixed_to_another_parameter:
            if (idx := parameters_fixp[i]) is None:
                raise TypeError(
                    f'An unsupported parameter p[{i}] is set to True in FixParams.'
                )
            index_fixp_target.append(i)
            index_fixp_source.append(int(idx))


def restore_params(params: tuple[float, ...]) -> tuple[float, ...]:
    '''Restore parameters with pfix by inserting parameters into params.
    - params -- parameter array. Its length is shorter than 14, which is the
                total number of parameters.
    '''
    global parameters_preset, index_free, index_fixp_target, index_fixp_source
    if (parameters_preset is None) or (len(params) == 14):
        return params
    parameters_preset[index_free] = params
    parameters_preset[index_fixp_target] = parameters_preset[index_fixp_source]
    return tuple(parameters_preset)


def shorten_init_and_bound_ifneeded(
    init: Sequence[float], bound: tuple[Sequence[float], Sequence[float]]
) -> tuple[tuple[float, ...], tuple[tuple[float, ...], tuple[float, ...]]]:
    '''Shorten init and bound parameter to match appropreate lengths.'''
    global index_free
    new_init = tuple(np.array(init)[index_free])
    new_bound0 = tuple(np.array(bound[0])[index_free])
    new_bound1 = tuple(np.array(bound[1])[index_free])
    return new_init, (new_bound0, new_bound1)
