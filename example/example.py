from tokult.core import Tokult
from tokult.fitting import get_bound_params, initialize_globalparameters, InputParams

tok = Tokult.launch(
    'cube_dirty.fits',
    'cube_dirty.psf.fits',
    ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
)
tok.set_region((226, 286), (226, 286), (5, 12))

init = tok.initialguess()
bound = get_bound_params(x0_dyn=(245, 265), y0_dyn=(245, 265))
sol = tok.imagefit(init=init, bound=bound, niter=8)

# init2 = InputParams(
#     x0_dyn=257.2535169953845,
#     y0_dyn=254.77380404948173,
#     PA_dyn=1.8559391327389154,
#     incliation_dyn=1.5285233939583551,
#     radius_dyn=0.15158789136535705,
#     velocity_sys=8.32404887449416,
#     mass_dyn=148.98759739843422,
#     brightness_center=0.2406206961350417,
#     velocity_dispersion=1.5,
#     radius_emi=1.2595011175728434,
#     x0_emi=257.4700751961959,
#     y0_emi=254.83584787950298,
#     PA_emi=1.9463569020823626,
#     inclination_emi=1.5345414507124888,
# )

# initialize_globalparameters(tok.datacube)
