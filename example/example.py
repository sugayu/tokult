import tokult
from tokult.core import Tokult
from tokult.fitting import (
    get_bound_params,
    initialize_globalparameters_for_image,
    InputParams,
)
from tokult.misc import fft2
import datetime
import numpy as np
from astropy.io import fits

# tok = Tokult.launch(
#     'cube_dirty.fits',
#     'cube_dirty.psf.fits',
#     ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
# )
# tok.set_region((226, 286), (226, 286), (5, 12))

# print(f'Start initialguess: {datetime.datetime.now()}')
# init = tok.initialguess()
# bound = get_bound_params(x0_dyn=(245, 265), y0_dyn=(245, 265), velocity_sys=(5, 12))
# print(f'Start imagefit: {datetime.datetime.now()}')
# sol_ls = tok.imagefit(init=init, bound=bound, optimization='ls')
# # sol_im = tok.imagefit(init=sol_ls.best, bound=bound, optimization='mc')
# print(f'End imagefit: {datetime.datetime.now()}')

tok_uv = Tokult.launch(
    'cube_dirty.fits',
    'cube_dirty.psf.fits',
    ('gamma1.fits', 'gamma2.fits', 'kappa.fits'),
)
tok_uv.set_region((226, 286), (226, 286), (5, 12))
hudl = fits.open('cube_dirty_uniform.psf.fits')
uvpsf_uniform = fft2(np.squeeze(hudl[0].data))
mask = (uvpsf_uniform[tok_uv.datacube.vslice, :, :]) > 1.3e-5

print(f'Start initialguess: {datetime.datetime.now()}')
init = tok_uv.initialguess()
bound = get_bound_params(
    x0_dyn=(245, 265),
    y0_dyn=(245, 265),
    PA_dyn=(0, 2 * np.pi),
    radius_dyn=(0.2, 5.0),
    velocity_sys=(5, 12),
    mass_dyn=(-2.0, 10.0),
    velocity_dispersion=(0.1, 3.0),
    brightness_center=(0.0, 100.0),
)
tok_uv.set_region((226 - 30, 286 + 30), (226 - 30, 286 + 30), (5, 12))
print(f'Start uvfit: {datetime.datetime.now()}')
sol_uv = tok_uv.uvfit(init=init, bound=bound, mask_for_fit=mask)
print(f'End uvfit: {datetime.datetime.now()}')

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

# initialize_globalparameters_for_image(tok.datacube)

# import corner

# sampler = sol_im.sampler
# flat_sample = sampler.get_chain(discard=2500, thin=600, flat=True)
# sample = sampler.get_chain()
# labels = sol_im.best._fields
# fig = corner.corner(flat_sample, labels=labels)
# plt.show()


# # make mask with uniform psf
# hudl = fits.open('cube_dirty_uniform.psf.fits')
# uvpsf_uniform = fft2(np.squeeze(hudl[0].data))
# # mask = np.log10(abs(uvpsf_uniform[:, :, :])) > 0.5
# mask = uvpsf_uniform.real > 1.3e-5
# uv = fft2(tok_uv.datacube.original)
# uvpsf = fft2(tok_uv.dirtybeam.original)
# uv_noise = uv / np.sqrt(uvpsf)
# for i in range(15):
#     plt.scatter(
#         uvpsf[i, :, :][mask[i, :, :]].real, uv_noise[i, :, :][mask[i, :, :]].real, s=1
#     )
#     plt.show()
# n = uv_noise[0, :, :][mask[0, :, :]].real
# p = uvpsf[0, :, :][mask[0, :, :]].real
# plt.hist(n[p > -p.min()], 50)
# plt.show()
# N = 1.0 / n[p > -p.min()].std() ** 2
# # d=np.squeeze(fits.getdata('/Volumes/SSD2TB_DATA/tokult/test/dv50_images_0.05arcsecpix/cube/Cy346_cube_dv50_dirty_natural.sumwt.fits'))
