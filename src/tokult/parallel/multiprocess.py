'''Multi process.
'''


def map_globals_to_childprocesses(pool: Pool) -> None:
    '''Map parent global parameters to parameters in pooled child processes.'''
    keys_globals = [
        'cube',
        'cube_error',
        'cubeshape',
        'cubeshape_imageplane',
        'xx_grid',
        'yy_grid',
        'vv_grid',
        'xslice',
        'yslice',
        'lensing',
        'lensing_interpolation',
        'convolve',
        'mask',
        # 'mask_FoV',
        'parameters_preset',
        'index_free',
        'index_fixp_target',
        'index_fixp_source',
    ]
    parent_globals = {}

    for k in keys_globals:
        try:
            parent_globals[k] = globals()[k]
        except KeyError:
            pass

    # fn_pkl = f'tokult_map_globals_to_childprocesses-{time.time()}.pkl'
    # with open(fn_pkl, 'wb') as f:
    #     pickle.dump(parent_globals, f)
    # NOTE: need to use private attributes to know # of processes used in pool.
    pool.map(_set_globals_in_process, [parent_globals] * pool._processes)  # type:ignore
    # Path(fn_pkl).unlink()


def _set_globals_in_process(parent_params: dict) -> None:
    '''Set global parameters in a child process.'''
    # with open(fn_pkl, 'rb') as f:
    #     parent_params = pickle.load(f)
    for key, value in parent_params.items():
        globals()[key] = value
