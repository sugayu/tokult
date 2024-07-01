def construct_convolvedmodel(params: tuple[float, ...]) -> np.ndarray:
    '''Construct a model detacube convolved with dirtybeam.'''
    global convolve
    model = construct_model_at_imageplane(params)
    model_convolved = convolve(model)
    # model_convolved = np.empty_like(model)
    # for i, image in enumerate(model):
    #     model_convolved[i, :, :] = convolve(image, index=i)
    return model_convolved
