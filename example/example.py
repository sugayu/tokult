import tokult
from tokult import Tokult
import datetime
import numpy as np
from astropy.io import fits

tok = Tokult.launch(
    'cube_dirty.fits',
    'cube_dirty.psf.fits',
    ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
)
tok.set_region((226, 286), (226, 286), (5, 12))
# tok.use_redshifts_of(z_lens=0.541, z_source=9.1111)
hudl = fits.open('cube_dirty_uniform.psf.fits')
uvpsf_uniform = tokult.misc.rfft2(np.squeeze(hudl[0].data))
mask = (uvpsf_uniform[tok.datacube.vslice, :, :]) > 1.3e-5

print(f'Start initialguess: {datetime.datetime.now()}')
init = tok.initialguess()
bound = tokult.get_bound_params(
    x0_dyn=(245, 265),
    y0_dyn=(245, 265),
    PA_dyn=(0, 2 * np.pi),
    radius_dyn=(0.2, 5.0),
    velocity_sys=(5, 12),
    mass_dyn=(-2.0, 10.0),
    velocity_dispersion=(0.1, 3.0),
    brightness_center=(0.0, 100.0),
)
print(f'Start uvfit: {datetime.datetime.now()}')
sol_uv = tok.uvfit(init=init, bound=bound, mask_for_fit=mask)
print(f'End uvfit: {datetime.datetime.now()}')
