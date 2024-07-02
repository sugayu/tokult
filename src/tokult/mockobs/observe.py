'''Mock observation pipeline.
'''

import numpy as np

__all__ = ['MockTelescope']


##
class MockTelescope:
    '''Telescope-like class to provide mock observations.'''

    def __init__(self) -> None: ...

    def observe(self, obj: np.ndarray) -> np.ndarray:
        '''Mock observation of the model object (image or cube).'''
        return obj
