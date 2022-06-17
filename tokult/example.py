'''example documents
'''
execfile('casa.py')
c.initialize_tokult()
c.fnames.readconfig_casa()
c.fnames.readconfig_gravlens()

conduct_listobs()
conduct_split()
imaging_cont()
imaging_cube()
imaging_cube_moments()
get_initparams()

write_spec_at_spaxels()
write_rms_at_ch()
fitgauss_spaxels()
write_mom0datatxt()
write_datacubetxt()
write_beamtxt()
write_gravlens()

param_m = fitting_mom0()
param_v = fitting_velocity2d(param_m)
fitting_cube3d(param_m, param_v)
