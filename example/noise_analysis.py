from tokult.core import Tokult
from tokult.fitting import (
    get_bound_params,
    initialize_globalparameters_for_image,
    InputParams,
)
import datetime

tok = Tokult.launch(
    'cube_dirty.fits',
    'cube_dirty.psf.fits',
    ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
)
tok.set_region((226, 286), (226, 286), (5, 12))

init = tok.initialguess()
bound = get_bound_params(x0_dyn=(245, 265), y0_dyn=(245, 265), velocity_sys=(5, 12))
sol_im = tok.imagefit(init=init, bound=bound, optimization='ls')

init_noise = sol_im.best
mock = tok.datacube.perturbed(convolve=tok.dirtybeam.fullconvolve)
tok_noise = Tokult.launch(
    mock,
    'cube_dirty.psf.fits',
    ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
    header_data=tok.datacube.header,
)
tok_noise.set_region((226, 286), (226, 286), (5, 12))
sol_noise = tok_noise.imagefit(init=init_noise, bound=bound, optimization='ls')
