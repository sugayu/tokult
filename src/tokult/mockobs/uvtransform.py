def construct_uvmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.'''
    global cubeshape, yslice, xslice
    model_cutout = construct_model_at_imageplane(params)
    model_image = np.zeros(cubeshape)
    model_image[:, yslice, xslice] = model_cutout
    # model_image = construct_model_at_imageplane(params)
    model_visibility = misc.rfft2(model_image)
    # image = misc.ifft2(model_visibility * beam_visibility)
    # return misc.fft2(image * mask_FoV)
    # return model_visibility * beam_visibility
    return model_visibility
