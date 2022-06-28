'''Modules to analyze measurement sets.
These modules are under construction.
'''
import os
import numpy as np
from pathlib import Path
from scipy.optimize import leastsq, least_squares
from scipy import signal
import scipy.special as sps
import astropy.io.fits as fits
from astropy import wcs
from . import common as c
from .casa_dummy import (
    listobs,
    split,
    tclean,
    imstat,
    immoments,
    exportfits,
    imfit,
    imhead,
    specflux,
    imsubimage,
)


##
def conduct_listobs() -> None:
    '''Conduct listobs to collect information
    '''
    if not os.path.exists(c.fnames.dname_project + c.fnames.mstxt):
        c.logger.debug('Conducting listobs...')
        listobs(vis=c.fnames.ms, listfile=c.fnames.dname_project + c.fnames.mstxt)
        c.logger.info(f'Save listobs of {c.fnames.ms} to {c.fnames.mstxt}.')


def conduct_split() -> None:
    '''Split original measurement sets.
    Probably this function will be not needed.
    '''
    lines_listobs = open(c.fnames.dname_project + c.fnames.mstxt).readlines()

    for i, line in enumerate(lines_listobs):
        if 'Spectral Windows:' in line:
            line_start = i
        if 'Sources:' in line:
            line_end = i
    num_spw = line_end - line_start - 2
    c.logger.debug(f'spw number ... {num_spw}')

    spec_spw_list = []
    for i in range(0, num_spw):
        info = lines_listobs[line_start + 2 + i].split()
        spw, ch = int(info[0]), int(info[2])
        if ch > 1:
            spec_spw_list.append(spw)
    spwtxt = ','.join([str(s) for s in spec_spw_list])

    c.logger.debug('Conducting split...')
    split(
        vis=c.fnames.ms,
        outputvis=c.fnames.dname_project + c.fnames.mssplit,
        spw=spwtxt,
        field=c.conf.field_id,
    )
    c.logger.info(f'Split {c.fnames.ms} into {c.fnames.mssplit}.')
    c.logger.debug('Conducting listobs...')
    listobs(
        vis=c.fnames.dname_project + c.fnames.mssplit,
        listfile=c.fnames.dname_project + c.fnames.mssplittxt,
    )
    c.logger.info(f'Save listobs of {c.fnames.mssplit} to {c.fnames.mssplittxt}.')


def get_spw() -> np.ndarray:
    '''Get spw including emission lines
    '''
    fname_splittxt = c.fnames.dname_project + c.fnames.mssplittxt
    lines_listobs_split = open(fname_splittxt).readlines()

    for i, line in enumerate(lines_listobs_split):
        if 'Spectral Windows:' in line:
            line_start = i
        if 'Sources:' in line:
            line_end = i

    num_spw_split = line_end - line_start - 2
    c.logger.debug(f'spw number split ... {num_spw_split}')

    spw = np.empty(num_spw_split)  # spwのIDのリストを作る
    ch0 = np.empty(num_spw_split)  # 一番最初のチャンネルの周波数（MHz）
    ch_width = np.empty(num_spw_split)  # １チャンネルの幅(khz)
    TotBW = np.empty(num_spw_split)  # spwの幅(kHz)
    Ctrfrq = np.empty(num_spw_split)  # spwの中心周波数(MHz)
    for i in range(0, num_spw_split):
        info = lines_listobs_split[line_start + 2 + i].split()
        spw[i] = int(info[0])
        ch0[i] = float(info[4])
        ch_width[i] = float(info[5])
        TotBW[i] = float(info[6])
        Ctrfrq[i] = float(info[7])

    assert isinstance(c.conf.nu_obs, str)
    assert isinstance(c.conf.nu_width, str)
    nu_obs = float(c.conf.nu_obs)
    nu_width = float(c.conf.nu_width)
    nu_start_obs = nu_obs - nu_width / 2
    nu_end_obs = nu_obs + nu_width / 2

    # ここでspwのはじめか終わりが中心から幅の範囲内にあるものを選ぶ(今回は1000km/sくらい)
    nu_start = Ctrfrq - TotBW * 1.0e-3 / 2 + np.abs(ch_width) * 1.0e-3 / 2
    nu_end = Ctrfrq + TotBW * 1.0e-3 / 2 - np.abs(ch_width) * 1.0e-3 / 2

    spw_selected = np.array([])
    for i in range(len(ch_width)):
        if (
            (nu_start_obs < nu_start[i] and nu_start[i] < nu_end_obs)
            or (nu_start_obs < nu_end[i] and nu_end[i] < nu_end_obs)
            or (nu_start[i] < nu_start_obs and nu_end_obs < nu_end[i])
        ):
            spw_selected = np.append(spw_selected, int(spw[i]))
    return spw_selected


def imaging_cont() -> None:
    '''Imaging continuum with tclean
    '''
    spw = get_spw()
    spw_selected = ','.join(spw.astype(int).astype(str))
    dname = c.fnames.dname_cont
    Path(dname).mkdir(exist_ok=True)
    params = {
        'vis': c.fnames.dname_project + c.fnames.mssplit,
        'field': c.conf.field_id_split,
        'spw': spw_selected,
        'specmode': 'mfs',
        'deconvolver': 'hogbom',
        'outframe': 'BARY',
        'nterms': 1,
        'imsize': [512, 512],
        'cell': ['0.05arcsec'],
        'weighting': c.conf.weight_cont,
        'pbcor': True,
        'interactive': False,
    }

    if not os.path.exists(dname + c.fnames.cont_dirty_img):
        imagename = dname + c.fnames.cont_dirty_img.replace('.image', '')
        c.logger.debug('Conducting tclean for dirty image...')
        tclean(imagename=imagename, niter=0, **params)
        c.logger.info(f'Create dirty image: {imagename}')

    if not os.path.exists(dname + c.fnames.cont_clean_img):
        imagename = dname + c.fnames.cont_clean_img.replace('.image', '')
        rms = imstat(dname + c.fnames.cont_dirty_img)['rms'][0]
        c.logger.debug('Conducting tclean  for clean image...')
        tclean(imagename=imagename, niter=10000, threshold=2 * rms, **params)
        c.logger.info(f'Create cleaned image: {imagename}')


def imaging_cube() -> None:
    '''Imaging line cube with tclean.
    '''

    spw = get_spw()
    spw_selected = ','.join(spw.astype(int).astype(str))

    # for type annotations
    assert isinstance(c.conf.dv, str)
    assert isinstance(c.conf.nu_obs, str)
    assert isinstance(c.conf.vel_start, str)
    assert isinstance(c.conf.chan_start, int)
    assert isinstance(c.conf.chan_end, int)

    # constants
    vel_width = c.conf.dv + 'km/s'
    restfq_in = c.conf.nu_obs + 'MHz'
    start_vel = c.conf.vel_start + 'km/s'

    cellsize = str(c.conf.pixsize) + 'arcsec'
    dname = c.fnames.dname_cube
    Path(dname).mkdir(exist_ok=True)
    params = {
        'vis': c.fnames.dname_project + c.fnames.mssplit,
        'field': c.conf.field_id_split,
        'spw': spw_selected,
        'specmode': 'cube',
        'deconvolver': 'hogbom',
        'width': vel_width,
        'start': start_vel,
        'outframe': 'BARY',
        'pbcor': True,
        'restfreq': restfq_in,
        'nterms': 1,
        'imsize': [c.conf.num_pix, c.conf.num_pix],
        'cell': [cellsize],
        'weighting': c.conf.weight_cube,
        'interactive': False,
    }

    cs, ce = c.conf.chan_start, c.conf.chan_end + 1
    fdirty = dname + c.fnames.cube_dirty_img
    fclean = dname + c.fnames.cube_clean_img

    # tclean
    if not os.path.exists(fdirty):
        imagename = fdirty.replace('.image', '')
        c.logger.debug('Conducting tclean for dirty cube...')
        tclean(imagename=imagename, niter=0, **params)
        c.logger.info(f'Create dirty cube: {imagename}')

    if not os.path.exists(fclean):
        rms = imstat(fdirty, axes=[0, 1])['rms'][cs:ce].mean()
        imagename = fclean.replace('.image', '')
        c.logger.debug('Conducting tclean for clean cube...')
        tclean(imagename=imagename, niter=10000, threshold=2.0 * rms, **params)
        c.logger.info(f'Create clean cube: {imagename}')


def imaging_cube_moments() -> None:
    '''Create moment maps with immoments
    '''
    # for type annotations
    assert isinstance(c.conf.chan_start, int)
    assert isinstance(c.conf.chan_end, int)

    chans = str(c.conf.chan_start) + '~' + str(c.conf.chan_end)
    rms_factor = c.conf.rms_factor  # S/Nがcube image上でこの値より大きいピクセルを使ってmoment mapを作る⭐️
    cs, ce = c.conf.chan_start, c.conf.chan_end + 1
    fdirty = c.fnames.dname_cube + c.fnames.cube_dirty_img
    fclean = c.fnames.dname_cube + c.fnames.cube_clean_img

    # immoment
    dname = c.fnames.dname_mom
    Path(dname).mkdir(exist_ok=True)
    rms_NB_dirty = imstat(fdirty, axes=[0, 1])['rms'][cs:ce].mean()
    rms_NB_clean = imstat(fclean, axes=[0, 1])['rms'][cs:ce].mean()

    def conduct_immoments(cubetype: str, fmom: str, mom: int = 0) -> None:
        '''Utility to conduct immommemts.
        '''
        if os.path.exists(dname + fmom):
            pass
        if cubetype == 'dirty':
            cube = fdirty
            rms = rms_NB_dirty
        elif cubetype == 'clean':
            cube = fclean
            rms = rms_NB_clean

        image = dname + fmom
        fitsimage = dname + fmom.replace('.image', '.fits')
        params = {'outfile': image, 'chans': chans, 'moments': mom}
        if mom == 0:
            immoments(cube, **params)
        elif mom in (1, 2):
            includepix = [rms_factor * rms, 100]
            immoments(cube, includepix=includepix, **params)
        exportfits(imagename=image, fitsimage=fitsimage)

    c.logger.debug('Conducting immoments...')
    conduct_immoments('dirty', fmom=c.fnames.line_dirty_mom0, mom=0)
    conduct_immoments('clean', fmom=c.fnames.line_clean_mom0, mom=0)
    conduct_immoments('dirty', fmom=c.fnames.line_dirty_mom1, mom=1)
    conduct_immoments('clean', fmom=c.fnames.line_clean_mom1, mom=1)
    conduct_immoments('dirty', fmom=c.fnames.line_dirty_mom1, mom=1)
    conduct_immoments('clean', fmom=c.fnames.line_clean_mom2, mom=2)
    c.logger.info('Create moment maps')


def get_initparams():
    '''Get initial parameters for ...?
    '''
    specfitsn = c.conf.specfitsn  # 速度mapを作る時のlinefitの基準、χ^2>specfitσ^2
    aparture_rad = c.conf.aparture_rad  # [arcsec]、3Dfitに使うspaxelの範囲、他の信号との分離
    mom0SN = c.conf.mom0SN  # mom0mapでこのSN以上のspaxelを使ってfitting
    ndivide = c.conf.ndivide  # aparture_rad/pixsize*2くらい、偶数
    kernel_num = c.conf.kernel_num  # beamfwhm(0.4")/pixsize*2程度、奇数
    angular_distance = c.conf.angular_distance
    rms_factor = c.conf.rms_factor

    dname = c.fnames.dname_mom
    Path(dname).mkdir(exist_ok=True)
    fmom0 = dname + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    header = imhead(fmom0)
    BMAJ = header['restoringbeam']['major']['value']
    BMIN = header['restoringbeam']['minor']['value']
    BPA = header['restoringbeam']['positionangle']['value']
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]
    c.logger.info(f'imfit centerpixel: ({Xcenter}, {Ycenter})')

    fname_init_para = c.fnames.dname_init_para + c.fnames.init_para
    Path(c.fnames.dname_init_para).mkdir(exist_ok=True)
    fdirty = c.fnames.dname_cube + c.fnames.cube_dirty_img
    cs, ce = c.conf.chan_start, c.conf.chan_end + 1
    rms_NB_dirty = imstat(fdirty, axes=[0, 1])['rms'][cs:ce].mean()

    with open(fname_init_para, 'w') as f:
        init_para = (
            f'{c.conf.dv} {c.conf.pixsize} {str(c.conf.num_pix)} {c.conf.vel_start} '
            f'{specfitsn} {aparture_rad} {mom0SN} {ndivide} '
            f'{kernel_num} {angular_distance} {str(c.conf.chan_start)} {str(c.conf.chan_end)} '
            f'{rms_factor} {Xcenter} {Ycenter} {rms_NB_dirty} '
            f'{rms_mom0} {BMAJ} {BMIN} {BPA} '
        )
        f.write(init_para)


def write_spec_at_spaxels():
    '''Write down spectra (velocity, flux) of each spaxel.
    '''

    image = c.fnames.dname_cube + c.fnames.cube_dirty_img
    # image1 = cube_dirty_name
    chans = str(c.conf.chan_start) + '~' + str(c.conf.chan_end)
    ngrid = c.conf.ngrid
    ndivide = c.conf.ndivide

    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    narray = np.arange(c.conf.ndivide + 1)
    xarray = round(Xcenter) - ngrid * ndivide / 2 + narray * ngrid
    yarray = round(Ycenter) + ngrid * ndivide / 2 - narray * ngrid
    # x1, y1 = np.meshgrid(xlist, ylist)

    dname = c.fnames.dname_spec
    Path(dname).mkdir(exist_ok=True)
    # for x2, y2 in zip(x1, y1):
    #     for x3, y3 in zip(x2, y2):
    for X in xarray:
        for Y in yarray:
            fspec = dname + str(round(X)) + "_" + str(round(Y)) + ".spectrum"
            fspec_out = fspec + ".txt"
            box = f'{str(X)},{str(Y)},{str(X)},{str(Y)}'
            # get spectra of a spaxel
            specflux(
                imagename=image, box=box, chans=chans, function='mean', logfile=fspec
            )
            os.system("awk 'NR>=5 {print $4,$5}' " + fspec + ">" + fspec_out)


def write_rms_at_ch():
    '''Write down rms at each channel.
    '''
    assert isinstance(c.conf.chan_start, int)
    assert isinstance(c.conf.chan_end, int)

    image = c.fnames.dname_cube + c.fnames.cube_dirty_img
    rmsname = c.fnames.dname_cube + c.fnames.rms
    rms = imstat(image, axes=[0, 1])['rms']
    with open(rmsname, 'w') as f:
        for i in range(c.conf.chan_start, c.conf.chan_end + 1):
            f.write(f'{rms[i]}\n')
    # IDEA
    # np.savetxt(rmsname, rms)


def func_lineGauss(x, params, consts):
    '''Gaussian function to be fitted to emission lines.
    '''
    p0, p1, p2 = params
    c0, c1 = consts
    # model_y = p0*xvalues + p1
    model_y = np.abs(p0) * np.exp(-0.5 * ((x - p1) / p2) ** 2.0) + c1
    # model_y = np.abs(p0)*np.exp(-0.5*((xvalues-p1)/p2)**2.) + c0*xvalues + c1
    return model_y


def calcchi(params, consts, model_func, x, y, ye):
    '''Calcurate chi for specGauss fitting.
    '''
    model = model_func(x, params, consts)
    chi = (y - model) / ye
    return chi


def solve_leastsq(x, y, ye, param_init, consts, model_func):
    '''Solve least chi square for specGauss fitting
    '''
    param_output = leastsq(
        calcchi, param_init, args=(consts, model_func, x, y, ye), full_output=True,
    )
    param_result, covar_output, info, mesg, ier = param_output
    error_result = np.sqrt(covar_output.diagonal())
    dof = len(x) - 1 - len(param_init)
    chi2 = np.sum(np.power(calcchi(param_result, consts, model_func, x, y, ye,), 2.0,))
    return [param_result, error_result, chi2, dof]


def fitgauss_spaxels():
    '''Fitting Gaussian to emission lines in spaxels
    '''
    fname_chi2 = c.fnames.dname_cube + c.fnames.chi2
    rms = np.loadtxt(c.fnames.dname_cube + c.fnames.rms, comments='!', unpack=True)

    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    assert isinstance(c.conf.dv, str)
    narray = np.arange(c.conf.ndivide + 1)
    xarray = round(Xcenter) - c.conf.ngrid * c.conf.ndivide / 2 + narray * c.conf.ngrid
    yarray = round(Ycenter) + c.conf.ngrid * c.conf.ndivide / 2 - narray * c.conf.ngrid

    dname = c.fnames.dname_spec
    with open(fname_chi2, 'w') as f:
        for x in xarray:
            for y in yarray:
                try:
                    fspec = f'{str(round(x))}_{str(round(y))}.spectrum.txt'
                    v, flux = np.loadtxt(dname + fspec, comments='!', unpack=True)

                    sigma_init = float(c.conf.dv)
                    norm_init = 1
                    init = np.array([norm_init, 0, sigma_init])
                    consts = np.array([0, 0])

                    result, error, chi2, dof = solve_leastsq(
                        v, flux, rms, init, consts, func_lineGauss
                    )
                    Vc, Vce = result[1], error[1]
                    sigma, sigmae = np.abs(result[2]), np.abs(error[2])
                    model_y = func_lineGauss(v, result, consts)
                    chi2 = sum(flux ** 2 / rms ** 2 - (flux - model_y) ** 2 / rms ** 2)
                    result = '{0} {1} {2} {3} {4} {5} {6}\n'.format(
                        x, y, chi2, Vc, Vce, sigma, sigmae
                    )
                    f.write(result)

                except:
                    ha = 1

    param = np.loadtxt(c.fnames.dname_cube + c.fnames.chi2, comments='!', unpack=True)
    x, y, chi2, V, Ve, sig, sige = param

    velname = c.fnames.dname_cube + c.fnames.velocity
    sigvname = c.fnames.dname_cube + c.fnames.sigma
    with open(velname, 'w') as fv, open(sigvname, 'w') as fs:
        for i in range(len(x)):
            if chi2[i] > c.conf.specfitsn ** 2:
                fv.write(f'{x[i]} {y[i]} {V[i]} {Ve[i]}\n')
                fs.write('{x[i]} {y[i]} {sig[i]} {sige[i]}\n')


def write_mom0datatxt():
    '''Write down mom0 data values in *.txt with highSN and area
    '''
    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    fmom0fits = c.fnames.dname_mom + c.fnames.line_dirty_mom0_fits
    sn_threshold = c.conf.mom0SN
    aparture_radius = c.conf.aparture_rad
    pixsize = float(c.conf.pixsize)

    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    rms_mom0 = imstat(fmom0)['rms'][0]
    hdul = fits.open(fmom0fits)
    data = np.squeeze(hdul[0].data)
    xshape_array = np.arange(data.shape[1])
    yshape_array = np.arange(data.shape[0])
    xx, yy = np.meshgrid(xshape_array, yshape_array)

    fname_mom = c.fnames.dname_mom + c.fnames.mom0_ch
    with open(fname_mom, 'w') as f:
        data_1d = data.ravel()
        xx_1d = xx.ravel()
        yy_1d = yy.ravel()
        for di, xi, yi in zip(data_1d, xx_1d, yy_1d):
            f.write(f'{xi} {yi} {di}\n')

    dist_thresh = (aparture_radius / pixsize) ** 2
    distance = (xx - Xcenter) ** 2 + (yy - Ycenter) ** 2
    mask_sn = data > sn_threshold * rms_mom0
    mask_dist = distance < dist_thresh

    mask = mask_sn & mask_dist
    data_highSN = data[mask]
    xx_highSN = xx[mask]
    yy_highSN = yy[mask]
    fname_highSN = c.fnames.dname_mom + c.fnames.mom0_highSN
    with open(fname_highSN, 'w') as f:
        for di, xi, yi in zip(data_highSN, xx_highSN, yy_highSN):
            f.write(f'{xi} {yi} {di}\n')

    fname_momdev = c.fnames.dname_mom + c.fnames.mom0_dev
    narray = np.arange(c.conf.ndivide + 1)
    xarray = round(Xcenter) - c.conf.ngrid * c.conf.ndivide / 2 + narray * c.conf.ngrid
    yarray = round(Ycenter) + c.conf.ngrid * c.conf.ndivide / 2 - narray * c.conf.ngrid

    with open(fname_momdev, 'w') as f:
        for x in xarray:
            for y in yarray:
                f.write(f'{x} {y} {data[int(y), int(x)]}\n')
    hdul.close()


def write_datacubetxt():
    '''Write down datacube into *.txt files
    '''
    assert isinstance(c.conf.chan_start, int)
    assert isinstance(c.conf.chan_end, int)

    fname_out = c.fnames.dname_project + c.fnames.output
    dname_spec = c.fnames.dname_spec
    sn_select_file = c.fnames.dname_mom + c.fnames.mom0_highSN
    x1, y1, flux = np.loadtxt(sn_select_file, comments='!', unpack=True)
    rms = np.loadtxt(c.fnames.dname_cube + c.fnames.rms, comments='!', unpack=True)
    with open(fname_out, 'w') as f:
        for x, y in zip(x1, y1):
            pixpos = str(round(x)) + "_" + str(round(y))
            fspec = pixpos + ".spectrum"
            fspectxt = fspec + ".txt"
            v, flux = np.loadtxt(dname_spec + fspectxt, comments='!', unpack=True)
            for i in range(c.conf.chan_end - c.conf.chan_start + 1):
                f.write(f'{x} {y} {v[i]} {flux[i]} {rms[i]}\n')

    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    cubefile = c.fnames.dname_project + c.fnames.cube_dev
    narray = np.arange(c.conf.ndivide + 1)
    xarray = round(Xcenter) - c.conf.ngrid * c.conf.ndivide / 2 + narray * c.conf.ngrid
    yarray = round(Ycenter) + c.conf.ngrid * c.conf.ndivide / 2 - narray * c.conf.ngrid

    with open(cubefile, 'w') as f:
        for x in xarray:
            for y in yarray:
                pixpos = str(round(x)) + "_" + str(round(y))
                fspec = pixpos + ".spectrum"
                fspectxt = fspec + ".txt"
                v, flux = np.loadtxt(dname_spec + fspectxt, comments='!', unpack=True)
                for i in range(c.conf.chan_end - c.conf.chan_start + 1):
                    f.write(f'{x} {y} {v[i]} {flux[i]} {rms[i]}\n')


def write_beamtxt():
    '''Write down beam data into *.txt files
    '''
    assert isinstance(c.conf.chan_start, int)
    assert isinstance(c.conf.chan_end, int)
    chanctr = str(int(np.round((c.conf.chan_start + c.conf.chan_end) / 2)))
    if not os.path.exists(c.fnames.dname_cube + c.fnames.beam):
        pfsimage = c.fnames.dname_cube + c.fnames.cube_dirty_img.replace(
            '.image', '.psf'
        )
        outfile = c.fnames.dname_cube + c.fnames.beam
        fitsimage = c.fnames.dname_cube + c.fnames.beam_fits
        imsubimage(imagename=pfsimage, outfile=outfile, chans=chanctr)
        exportfits(imagename=outfile, fitsimage=fitsimage)

    fname_beamfits = c.fnames.dname_cube + c.fnames.beam_fits
    hdul = fits.open(fname_beamfits)
    hdu_beam = hdul[0]
    data = hdu_beam.data
    data = np.squeeze(data)
    xshape_array = np.arange(data.shape[0])
    yshape_array = np.arange(data.shape[1])
    xx, yy = np.meshgrid(xshape_array, yshape_array)
    fname_beamdat = c.fnames.dname_cube + c.fnames.beam_dat
    with open(fname_beamdat, 'w') as f:
        for di, xi, yi in zip(data.ravel(), xx.ravel(), yy.ravel()):
            f.write(f'{xi} {yi} {di}\n')
    hdul.close()


def write_gravlens():
    '''Write down gravlens infomation into *.txt files
    '''
    imagename1 = c.fnames.dname_mom + c.fnames.line_dirty_mom0_fits
    hdul = fits.open(imagename1)
    wcs_data = wcs.WCS(hdul[0].header)

    hdul_gamma1 = fits.open(c.conf.fname_gamma1)
    wcs_gl = wcs.WCS(hdul_gamma1[0].header)
    gamma1 = np.squeeze(hdul_gamma1[0].data)

    hdul_gamma2 = fits.open(c.conf.fname_gamma2)
    gamma2 = np.squeeze(hdul_gamma2[0].data)

    hdul_kappa = fits.open(c.conf.fname_kappa)
    kappa = np.squeeze(hdul_kappa[0].data)

    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    xlow = np.round(Xcenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    xup = np.round(Xcenter + c.conf.ndivide / 2 + c.conf.kernel_num)
    ylow = np.round(Ycenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    yup = np.round(Ycenter + c.conf.ndivide / 2 + c.conf.kernel_num)

    dname = c.fnames.dname_gl
    Path(dname).mkdir(exist_ok=True)
    fname_gl = dname + c.fnames.gravlens
    with open(fname_gl, 'w') as f:
        for x_pix in range(int(xlow), int(xup + 1)):
            for y_pix in range(int(ylow), int(yup + 1)):
                coord_wcs = wcs_data.wcs_pix2world(x_pix, y_pix, 0, 0, 0)
                coord_pix_gl = wcs_gl.wcs_world2pix(coord_wcs[0], coord_wcs[1], 0)
                xi = int(np.round(coord_pix_gl[0]))
                yi = int(np.round(coord_pix_gl[1]))
                f.write(
                    f'{x_pix}  {y_pix}  {gamma1[yi,xi]}  {gamma2[yi,xi]}  {kappa[yi,xi]}\n'
                )
    hdul.close()
    hdul_gamma1.close()
    hdul_gamma2.close()
    hdul_kappa.close()


def glensing(
    pos: np.ndarray, g1: np.ndarray, g2: np.ndarray, k: np.ndarray
) -> np.ndarray:
    '''Convert coordinates with gravitational lensing from image plane to source plane.
    Keyword Arguments:
    pos -- position array including (x, y). shape: (n, m, 2) and shape of x: (n, m)
    g1, g2, k -- gamma1, gamma2, kappa array. shape: (n, m)
    '''
    jacob = np.array([[1 - k - g1, -g2], [-g2, 1 - k + g1]])
    axis = np.concatenate((2 + np.arange(g1.ndim), (0, 1)))
    # assert axis == np.array([2, 3, 0, 1])
    jacob = jacob.transpose(axis)
    # assert jacob.shape == (n, m, 2, 2)
    _pos = pos[..., np.newaxis]
    return np.squeeze(jacob @ _pos)


def rotate_coord(pos: np.ndarray, angle: np.ndarray) -> np.ndarray:
    '''Rotate (x,y) coordinates
    Keyword Arguments:
    pos -- position array. shape: (n, m, 2)
    angle -- scalar; angle to rotate. radian
    '''
    rot = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    _pos = pos[..., np.newaxis]
    return np.squeeze(rot @ _pos)


def beam_convolve(flux, Sbeam):
    '''Convolvep
    '''
    s1 = np.arange(c.conf.kernel_num)
    t1 = np.arange(c.conf.kernel_num)
    s2, t2 = np.meshgrid(s1, t1)
    s3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + s2
    t3 = c.conf.num_pix / 2 - (c.conf.kernel_num - 1) / 2 + t2
    st = np.array(c.conf.num_pix * s3 + t3, dtype=int)
    kernel = Sbeam[st]
    kernel2 = kernel / sum(sum(kernel))
    return signal.convolve2d(flux, kernel2, 'same')


def polar_coordinate(x, y):
    '''Convert (x, y) to polar coordinates (r, phi)
    '''
    r = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return r, phi


def fitting_mom0():
    '''Fitting moment0 map.
    '''
    fname_beamdat = c.fnames.dname_cube + c.fnames.beam_dat
    a, b, Sbeam = np.loadtxt(fname_beamdat, comments='!', unpack=True)

    fname_mom0 = c.fnames.dname_mom + c.fnames.mom0_highSN
    x, y, f = np.loadtxt(fname_mom0, comments='!', unpack=True)
    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(c.fnames.dname_mom + c.fnames.line_dirty_mom0)['rms'][0]
    fe = rms_mom0

    fname_gl = c.fnames.dname_gl + c.fnames.gravlens
    xgl, ygl, gamma1, gamma2, kappa = np.loadtxt(fname_gl, comments='!', unpack=True)

    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    xlow = np.round(Xcenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    xup = np.round(Xcenter + c.conf.ndivide / 2 + c.conf.kernel_num)
    ylow = np.round(Ycenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    yup = np.round(Ycenter + c.conf.ndivide / 2 + c.conf.kernel_num)

    xarray = np.arange(xlow, xup + 1, 1)
    yarray = np.arange(ylow, yup + 1, 1)
    xx_grid, yy_grid = np.meshgrid(xarray, yarray)

    param = [Xcenter, Ycenter, np.pi / 2, 0.25 * np.pi, 1.0, 1.0]
    param_bounds = (
        (-np.inf, -np.inf, 0, 0, 0, 0),
        (np.inf, np.inf, np.pi, 0.5 * np.pi, np.inf, np.inf),
    )

    def fitting_func_mom0(p, x, y, f, fe):
        idx = ((yup - ylow + 1) * (xx_grid - xlow) + yy_grid - ylow).astype(int)
        g1 = gamma1[idx]
        g2 = gamma2[idx]
        k = kappa[idx]
        pos_grid = np.array([(xx_grid - p[0]), (yy_grid - p[1])]).transpose(1, 2, 0)
        pos_grid = glensing(pos_grid, g1, g2, k)
        pos_grid = rotate_coord(pos_grid, -p[2])
        xx, yy = pos_grid.transpose(2, 0, 1)

        yy = yy / np.cos(p[3])  # inclination
        rr, _ = polar_coordinate(xx, yy)
        flux_density = p[5] * np.exp(-rr / p[4])

        model_grid = beam_convolve(flux_density, Sbeam)
        ix = (x - xlow).astype(int)
        iy = (y - ylow).astype(int)
        residual = (f - model_grid[iy, ix]) / fe
        return residual

    optimised_param = least_squares(
        fitting_func_mom0, param, args=(x, y, f, fe), bounds=param_bounds
    )

    xctrO = optimised_param['x'][0]
    yctrO = optimised_param['x'][1]
    thetaO = optimised_param['x'][2]
    incO = optimised_param['x'][3]
    hO = optimised_param['x'][4]
    gO = optimised_param['x'][5]

    print("i[°]:", incO / np.pi * 180)
    print("h[pix]:", hO)
    # print("h[kpc]:", hO * pixkpc)
    print("theta[°]:", thetaO / np.pi * 180)
    print("centerpixel:[", xctrO, yctrO, "]")
    print("center flux density:", gO)
    print('mom0fit done..')
    return optimised_param['x']


def get_init_fitting_velocity2d(param):
    '''
    Keyword Arguments:
    param -- best-fit parameter of moment0 map fitting
    '''
    xctrO = param[0]
    yctrO = param[1]
    thetaO = param[2]
    incO = param[3]
    hO = param[4]
    # gO = param[5]
    init = [incO, hO, 100, 0, thetaO, xctrO, yctrO]
    return init


def fitting_velocity2d(param_fit_mom0):
    '''Fitting 2d velocity map.
    param_fit_mom0 -- best-fit parameter of moment0 map fitting
    '''
    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    xlow = np.round(Xcenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    xup = np.round(Xcenter + c.conf.ndivide / 2 + c.conf.kernel_num)
    ylow = np.round(Ycenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    yup = np.round(Ycenter + c.conf.ndivide / 2 + c.conf.kernel_num)

    xarray = np.arange(xlow, xup + 1, 1)
    yarray = np.arange(ylow, yup + 1, 1)
    xx_grid, yy_grid = np.meshgrid(xarray, yarray)

    fname_beamdat = c.fnames.dname_cube + c.fnames.beam_dat
    xbeam, ybeam, Sbeam = np.loadtxt(fname_beamdat, comments='!', unpack=True)

    fname_vel = c.fnames.dname_cube + c.fnames.velocity
    xv, yv, Vs, Vse = np.loadtxt(fname_vel, comments='!', unpack=True)

    c.fnames_gl = c.fnames.dname_gl + c.fnames.gravlens
    xgl, ygl, gamma1, gamma2, kappa = np.loadtxt(c.fnames_gl, comments='!', unpack=True)
    G = 1

    def fitting_func_vfield(p, x, y, V, Ve):
        idx = ((yup - ylow + 1) * (xx_grid - xlow) + yy_grid - ylow).astype(int)
        g1 = gamma1[idx]
        g2 = gamma2[idx]
        k = kappa[idx]
        pos_grid = np.array([xx_grid - p[5], yy_grid - p[6]]).transpose(1, 2, 0)
        pos_grid = glensing(pos_grid, g1, g2, k)
        pos_grid = rotate_coord(pos_grid, -p[4])
        xx, yy = pos_grid.transpose(2, 0, 1)

        yy = yy / np.cos(p[0])  # inclination
        rr, pphi = polar_coordinate(xx, yy)
        r2h = 0.5 * rr / p[1]

        I_0 = sps.iv(0, r2h)
        I_1 = sps.iv(1, r2h)
        K_0 = sps.kv(0, r2h)
        K_1 = sps.kv(1, r2h)
        A = I_0 * K_0 - I_1 * K_1
        f_sightline = np.cos(pphi) * np.sin(p[0])
        vel_r = p[3] + np.sqrt(4 * np.pi * G * p[2] * p[1] * r2h ** 2 * A) * f_sightline

        # convolve velocity?
        model_vel_grid = beam_convolve(vel_r, Sbeam)
        ix = (x - xlow).astype(int)
        iy = (y - ylow).astype(int)
        residual = (V - model_vel_grid[iy, ix]) / Ve
        return residual

    init = get_init_fitting_velocity2d(param_fit_mom0)
    param_bounds = (
        (0, 0, 0, -np.inf, 0, -np.inf, -np.inf),
        (0.5 * np.pi, np.inf, np.inf, np.inf, 2 * np.pi, np.inf, np.inf),
    )

    optimised_param = least_squares(
        fitting_func_vfield, init, args=(xv, yv, Vs, Vse), bounds=param_bounds
    )

    i2db = optimised_param['x'][0]
    h2db = optimised_param['x'][1]
    myu_02db = optimised_param['x'][2]
    Vsys2db = optimised_param['x'][3]
    theta2db = optimised_param['x'][4]
    xctr2db = optimised_param['x'][5]
    yctr2db = optimised_param['x'][6]

    mpix = c.conf.mpix  # XXX
    mdisk = (
        myu_02db
        * 2
        * np.pi
        * c.const.kgMo
        * 1e6
        * (h2db ** 2)
        / (mpix * c.const.g)
        / 1e9
    )
    print("i[°]:" + '{:.1f}'.format(np.rad2deg(i2db)))
    print("h[pix]:" + '{:.2f}'.format(h2db))
    print("Mdisk[10^9Msolar]:" + '{:.2f}'.format(mdisk))
    print("Vsys[km/s]:" + '{:.1f}'.format(Vsys2db))
    print("theta[°]:" + '{:.1f}'.format(np.rad2deg(theta2db)))
    print("[xctrb,yctrb]:[", xctr2db, yctr2db, "]")
    print('velfit2d done..')

    return optimised_param['x']


def func_Gauss(x, center, sigma, area):
    '''Gaussian function.
    '''
    norm = area / (np.sqrt(2 * np.pi) * sigma)
    return norm * np.exp(-((x - center) ** 2) / (2.0 * sigma ** 2))


def get_init_fitting3d(param_m, param_v):
    '''
    Keyword Arguments:
    param_m -- best-fit parameter of moment0 map fitting
    param_v -- best-fit parameter of velocity map fitting
    '''
    xctrO = param_m[0]
    yctrO = param_m[1]
    thetaO = param_m[2]
    incO = param_m[3]
    hO = param_m[4]
    gO = param_m[5]

    i2db = param_v[0]
    h2db = param_v[1]
    myu_02db = param_v[2]
    Vsys2db = param_v[3]
    theta2db = param_v[4]
    xctr2db = param_v[5]
    yctr2db = param_v[6]

    return [
        xctr2db,
        yctr2db,
        theta2db,
        i2db,
        h2db,
        Vsys2db,
        2 * np.pi * myu_02db * h2db ** 2,
        gO,
        float(c.conf.dv),
        hO,
        xctrO,
        yctrO,
        thetaO,
        incO,
    ]


def fitting_cube3d(param_m, param_v):
    '''Fitting data cube.
    Keyword Arguments:
    param_m -- best-fit parameter of moment0 map fitting
    param_v -- best-fit parameter of velocity map fitting
    '''

    fname_beamdat = c.fnames.dname_cube + c.fnames.beam_dat
    _, _, Sbeam = np.loadtxt(fname_beamdat, comments='!', unpack=True)
    fname_gl = c.fnames.dname_gl + c.fnames.gravlens
    xgl, ygl, gamma1, gamma2, kappa = np.loadtxt(fname_gl, comments='!', unpack=True)

    fmom0 = c.fnames.dname_mom + c.fnames.line_dirty_mom0
    rms_mom0 = imstat(imagename=fmom0)['rms'][0]
    res_imfit = imfit(imagename=fmom0, region=c.conf.region, rms=rms_mom0)
    Xcenter = res_imfit['results']['component0']['pixelcoords'][0]
    Ycenter = res_imfit['results']['component0']['pixelcoords'][1]

    xlow = np.round(Xcenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    xup = np.round(Xcenter + c.conf.ndivide / 2 + c.conf.kernel_num)
    ylow = np.round(Ycenter - c.conf.ndivide / 2 - c.conf.kernel_num)
    yup = np.round(Ycenter + c.conf.ndivide / 2 + c.conf.kernel_num)

    vel_start = int(c.conf.vel_start)
    dv = int(c.conf.dv)
    xarray = np.arange(xlow, xup + 1, 1)
    yarray = np.arange(ylow, yup + 1, 1)
    vstart = vel_start + dv * c.conf.chan_start
    vend = vel_start + dv * c.conf.chan_end + 1
    varray = np.arange(vstart, vend, dv)
    xx_grid, yy_grid, vv_grid = np.meshgrid(xarray, yarray, varray)
    idx = np.array((yup - ylow + 1) * (xx_grid - xlow) + yy_grid - ylow, dtype=int)
    g1 = gamma1[idx]
    g2 = gamma2[idx]
    k = kappa[idx]

    def solve_least_squares(
        x, y, V, I, Ie, param_init, model_func, param_bounds,
    ):
        output = least_squares(
            get_chi, param_init, args=(model_func, x, y, V, I, Ie), bounds=param_bounds,
        )
        param_bestfit = output.x
        dof = len(x) - 1 - len(param_init)
        chi2 = np.sum(np.power(get_chi(param_bestfit, model_func, x, y, V, I, Ie), 2.0))
        J = output.jac
        # residuals_lsq = Ivalues - model_func(xvalues, yvalues, Vvalues, param_result)
        cov = np.linalg.inv(J.T.dot(J))  # * (residuals_lsq**2).mean()
        result_error = np.sqrt(np.diag(cov))
        return [param_bestfit, result_error, chi2, dof, cov]

    def get_chi(params, model_func, x, y, V, I, Ie):
        model = model_func(x, y, V, params)
        chi = (I - model) / Ie
        return chi

    def Imodel(x, y, v, params):
        p0, p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13 = params

        # velocity field
        pos_grid = np.array([xx_grid - p0, yy_grid - p1]).transpose(1, 2, 3, 0)
        pos_grid = glensing(pos_grid, g1, g2, k)
        pos_grid = rotate_coord(pos_grid, -p2)
        xx, yy = pos_grid.transpose(3, 0, 1, 2)
        yy = yy / np.cos(p3)  # inclination
        rr, pphi = polar_coordinate(xx, yy)
        r2h = 0.5 * rr / p4
        myu_0 = p6 / (2 * np.pi * p4 ** 2)
        G = 1
        A = sps.iv(0, r2h) * sps.kv(0, r2h) - sps.iv(1, r2h) * sps.kv(1, r2h)
        f_sightline = np.cos(pphi) * np.sin(p3)
        velocity = p5 + np.sqrt(4 * np.pi * G * myu_0 * p4 * r2h ** 2 * A) * f_sightline

        # spatial flux
        pos_grid = np.array([xx_grid - p10, yy_grid - p11]).transpose(1, 2, 3, 0)
        pos_grid = glensing(pos_grid, g1, g2, k)
        pos_grid = rotate_coord(pos_grid, -p12)
        xx_f, yy_f = pos_grid.transpose(3, 0, 1, 2)
        yy_f = yy_f / np.cos(p13)
        rr_f, _ = polar_coordinate(xx_f, yy_f)
        flux = p7 * np.exp(-rr_f / p9)

        # convolvolution
        model = func_Gauss(vv_grid, center=velocity, sigma=p8, area=flux)
        model2 = np.zeros(xx_grid.shape)
        for i in range(xx_grid.shape[2]):
            model2[:, :, i] = beam_convolve(model[:, :, i], Sbeam)
        iv = (v - dv * c.conf.chan_start - vel_start) / dv
        iv = np.round(iv).astype(int)
        ix = (x - xlow).astype(int)
        iy = (y - ylow).astype(int)
        Imodel_last = model2[iy, ix, iv]
        return Imodel_last

    fname_cube = c.fnames.dname_project + c.fnames.output
    xi, yi, Vi, I, Ie = np.loadtxt(fname_cube, comments='!', unpack=True)
    init = get_init_fitting3d(param_m, param_v)
    inf, pi = np.inf, np.pi
    bounds_param = (
        (-inf, -inf, 0, 0, 0, -inf, 0, 0, 0, 0, -inf, -inf, 0, 0),
        (inf, inf, 2 * pi, pi / 2, inf, inf, inf, inf, inf, inf, inf, inf, pi, pi / 2,),
    )

    # 01中心座標b、2PAb、3傾斜角b,4スケール長b,5システム速度,6中心面密度,7中心強度,8速度分散,9スケール長O,1011中心座標O,12PAO
    # result=[xb,yb,PAb,i,hb,Vsys,myu_0,gO,sigma,hO,xO,yO,PAO]
    c.logger.debug('Fitting...')
    result, resulte, chi2, dof, cov = solve_least_squares(
        xi, yi, Vi, I, Ie, init, Imodel, bounds_param
    )
    fname_result = c.fnames.dname_result + c.fnames.bestfit
    Path(c.fnames.dname_result).mkdir(exist_ok=True)
    np.savetxt(fname_result, result)
    c.logger.info(f'Save results in {fname_result}')

    printout_results(result, resulte, chi2, dof, cov)


def printout_results(result, resulte, chi2, dof, cov):
    '''Printout results with simple units.
    Keyword Arguments:
    result  -- best-fit results
    resulte -- best-fit result errors
    '''
    # 残差出して、誤差求める
    xctr3db = result[0]
    yctr3db = result[1]
    theta3db = result[2]
    i3db = result[3]
    h3db = result[4]
    Vsys3d = result[5]
    Mdisk3d = result[6]
    g3dO = result[7]
    sigma3dO = result[8]
    h3dO = result[9]
    xctr3dO = result[10]
    yctr3dO = result[11]
    theta3dO = result[12]
    i3dO = result[13]

    xctr3dbe = resulte[0]
    yctr3dbe = resulte[1]
    theta3dbe = resulte[2]
    i3dbe = resulte[3]
    h3dbe = resulte[4]
    Vsys3de = resulte[5]
    Mdisk3de = resulte[6]
    g3dOe = resulte[7]
    sigma3dOe = resulte[8]
    h3dOe = resulte[9]
    xctr3dOe = resulte[10]
    yctr3dOe = resulte[11]
    theta3dOe = resulte[12]
    i3dOe = resulte[13]

    mpix = c.conf.mpix  # XXX
    Vrot = 0.88 * np.sqrt((Mdisk3d * 1.0e6) / (2 * h3db)) / 1000
    Vrote = Vrot * (
        (-h3dbe / h3db / 2) ** 2
        + (Mdisk3de / Mdisk3d / 2) ** 2
        + 2 * cov[4, 6] * (-1 / h3db / 2) * (1 / Mdisk3d / 2)
    ) ** (0.5)
    Vrot_sigma = Vrot / sigma3dO
    dia = (
        (-sigma3dOe / sigma3dO) ** 2
        + (Mdisk3de / Mdisk3d / 2) ** 2
        + (h3dbe / h3db / 2) ** 2
    )
    sigma_M = cov[6, 8]
    sigma_h = cov[8, 4]
    M_h = cov[4, 6]
    non_dia = (
        -sigma_M / sigma3dO / Mdisk3d
        + sigma_h / sigma3dO / h3db
        - M_h / Mdisk3d / h3db / 2
    )
    Vrot_sigmae = Vrot_sigma * np.sqrt(dia + non_dia)

    radtopi = 180 / np.pi
    print('reduced chi square:', chi2 / dof)
    print('Vrot_sigma:{0:.2f}+-{1:.2f}'.format(Vrot_sigma, Vrot_sigmae))
    print('Vrot:{0:.2f}+-{1:.2f}'.format(Vrot, Vrote))
    print('incl b[°]:{0:.1f}+-{1:.1f}'.format(i3db * radtopi, i3dbe * radtopi))
    print('incl O[°]:{0:.1f}+-{1:.1f}'.format(i3dO * radtopi, i3dOe * radtopi))
    print('theta b[°]:{0:.1f}+-{1:.1f}'.format(theta3db * radtopi, theta3dbe * radtopi))
    print('theta O[°]:{0:.1f}+-{1:.1f}'.format(theta3dO * radtopi, theta3dOe * radtopi))
    print(
        'disk mass[10^9Mo]:{0:.2f}+-{1:.2f}'.format(
            Mdisk3d * c.const.kgMo * (10 ** 6) / (mpix * (6.67 * 10 ** -11)) / 10 ** 9,
            Mdisk3de * c.const.kgMo * (10 ** 6) / (mpix * (6.67 * 10 ** -11)) / 10 ** 9,
        )
    )
    print('hb[pix]:{0:.2f}+-{1:.2f}'.format(h3db, h3dbe))
    print('hO[pix]:{0:.2f}+-{1:.2f}'.format(h3dO, h3dOe))
    print('Velsys[km/s]:{0:.1f}+-{1:.1f}'.format(Vsys3d, Vsys3de))
    print('center flux density[Jy/beam]:{0:.2f}+-{1:.2f}'.format(g3dO, g3dOe))
    print('vel sigma[km/s]:{0:.1f}+-{1:.1f}'.format(sigma3dO, sigma3dOe))
    print(
        'center pixelb:[{0:.1f}+-{1:.1f},{2:.1f}+-{3:.1f}]'.format(
            xctr3db, xctr3dbe, yctr3db, yctr3dbe
        )
    )
    print(
        'center pixelO:[{0:.1f}+-{1:.1f},{2:.1f}+-{3:.1f}]'.format(
            xctr3dO, xctr3dOe, yctr3dO, yctr3dOe
        )
    )
    print('3Dfit  done')
