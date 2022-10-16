import tokult
import datetime
import numpy as np
from astropy.io import fits
import multiprocessing

tok = tokult.Tokult.launch(
    'cube_dirty.fits',
    'cube_dirty.psf.fits',
    ('x-arcsec-deflect.fits', 'y-arcsec-deflect.fits'),
)
tok.use_region((226, 286), (226, 286), (5, 12))
tok.use_redshifts(z_source=9.1096, z_lens=0.541)

hudl = fits.open('cube_dirty_uniform.psf.fits')
uvpsf_uniform = tokult.misc.rfft2(np.squeeze(hudl[0].data))
uvcoverage = (uvpsf_uniform[tok.datacube.vslice, :, :]) > 1.0e-5

print(f'Start initialguess: {datetime.datetime.now()}')
init = tok.initialguess()
init = init._replace(mass_dyn=2.0)
bound = tokult.get_bound_params(
    x0_dyn=(245, 265),
    y0_dyn=(245, 265),
    PA_dyn=(0, 2 * np.pi),
    radius_dyn=(0.01, 5.0),
    velocity_sys=(5, 12),
    mass_dyn=(-5.0, 3.0),
    velocity_dispersion=(0.1, 3.0),
    brightness_center=(0.0, 1.0),
)
print(f'Start uvfit: {datetime.datetime.now()}')
with multiprocessing.Pool(4) as pool:
    sol = tok.uvfit(
        init=init,
        bound=bound,
        mask_for_fit=uvcoverage,
        pool=pool,
        progressbar=True,
        nwalkers=64,
        nsteps=8000,
    )
print(f'End uvfit: {datetime.datetime.now()}')
