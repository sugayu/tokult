'''Dummy CASA modules
'''


def listobs(vis, listfile):
    '''Dummy function of listobs.
    '''
    pass


def split(vis, outputvis, spw, field):
    '''Dummy function of split.
    '''
    pass


def tclean(
    vis,
    imagename,
    field,
    spw,
    specmode,
    deconvolver,
    outframe,
    nterms,
    imsize,
    cell,
    weighting,
    niter,
    pbcor,
    interactive,
    threshold=None,
):
    '''Dummy function of tclean.
    '''
    pass


def imstat(imagename, axes=None):
    '''Dummy function of imstat.
    '''
    return 0


def immoments(fname, outfile, chans, moments, includepix=None):
    '''Dummy function of imstat.
    '''
    pass


def exportfits(imagename, fitsimage):
    '''Dummy function of exportfits.
    '''
    pass


def imfit(imagename, region, rms):
    '''Dummy function of exportfits.
    '''
    pass


def imhead(name):
    '''Dummy function of imhead.
    '''
    pass


def specflux(imagename, box, chans, function, logfile):
    '''Dummy function of specflux.
    '''
    pass


def imsubimage(imagename, outfile, chans):
    '''Dummy function of imsubimage.
    '''
    pass
