'''Shared Memory provides us with variables sharing the memory over processes.

This function would be helpful to reduce a memory size in multi-process fitting.
'''

import os
from copy import deepcopy
from multiprocessing import shared_memory
from multiprocessing.managers import SharedMemoryManager
import numpy as np
from astropy.nddata import NDData, NDUncertainty
from logging import getLogger

logger = getLogger(__name__)
__all__ = ['SharedMemoryNDData']


##
class SharedMemoryNDData(NDData):
    '''SharedMemory wrapper of NDData.

    Example:
        shape = (100, 100)
        nddata = NDData(np.empty(shape), uncertainty=np.ones(shape))
        with SharedMemoryManager() as smm:
            nddata_shared = SharedMemoryNDData(nddata, smmanager=smm)
    '''

    def __init__(self, *args, smmanager: SharedMemoryManager | None = None) -> None:
        if smmanager is None:
            raise ValueError('SharedMemoryManager must be input.')
        self.smm = smmanager
        super().__init__(*args)
        self._data_shm: SharedNDArray
        self._mask_shm: SharedNDArray | None
        self._uncertainty_shm: SharedNDArray
        self._uncertainty_template: NDUncertainty
        self._uncertainty_cache: NDUncertainty | None = None

    def close(self) -> None:
        self._data_shm.close()
        if self._mask_shm is not None:
            self._mask_shm.close()
        self._uncertainty_shm.close()

    @property
    def is_subprocess(self) -> bool:
        '''Check if it is a subprocess where the object is created by pickle.'''
        return self.smm is None

    @property
    def _data(self) -> np.ndarray:
        return self._data_shm.data

    @_data.setter
    def _data(self, data: np.ndarray) -> None:
        if self.is_subprocess:
            raise ValueError('Attributes cannot be set in a subprocess.')
        self._data_shm = SharedNDArray(data, self.smm)

    @property
    def _mask(self) -> np.ndarray | None:
        if self._mask_shm is None:
            return None
        return self._mask_shm.data

    @_mask.setter
    def _mask(self, data: np.ndarray | None) -> None:
        if data is None:
            self._mask_shm = None
            return
        if self.is_subprocess:
            raise ValueError('Attributes cannot be set in a subprocess.')
        self._mask_shm = SharedNDArray(data, self.smm)

    @property
    def _uncertainty(self) -> NDUncertainty:
        if self._uncertainty_cache is None:
            self._uncertainty_cache = deepcopy(self._uncertainty_template)
            self._uncertainty_cache.array = self._uncertainty_shm.data
        return self._uncertainty_cache

    @_uncertainty.setter
    def _uncertainty(self, uncertainty: NDUncertainty) -> None:
        if self.is_subprocess:
            raise ValueError('Attributes cannot be set in a subprocess.')
        self._uncertainty_shm = SharedNDArray(uncertainty.array, self.smm)
        uncertainty.array = np.array([])
        self._uncertainty_template = uncertainty
        self._uncertainty_cache = None

    def __getstate__(self) -> dict:
        _dict = self.__dict__
        _dict['smm'] = None
        _dict['_uncertainty_cache'] = None
        return _dict


class SharedNDArray:
    '''Shared numpy.ndarray.'''

    def __init__(self, data: np.ndarray, smm: SharedMemoryManager) -> None:
        shm = smm.SharedMemory(size=data.nbytes)
        logger.info(f'Shared memory {shm.name} is created.')

        data_tmp: np.ndarray = np.ndarray(data.shape, data.dtype, buffer=shm.buf)
        data_tmp[:] = data[:]
        self.shape = data.shape
        self.dtype = data.dtype
        self._name = shm.name
        self._data = data_tmp
        self.shm = shm

    @property
    def data(self) -> np.ndarray:
        try:
            return self._data
        except AttributeError:
            self._data = np.ndarray(self.shape, self.dtype, buffer=self.shm.buf)
            logger.info(f'Data shared in {self._name} are cached at pid {os.getpid()}.')
            return self._data

    # # Unlink will be managed by SharedMemoryManager
    # def unlink(self) -> None:
    #     self.shm = shared_memory.SharedMemory(name=self._name)
    #     try:
    #         self.shm.close()
    #         self.shm.unlink()
    #         logger.info(f'Shared memory {self._name} is unlinked.')
    #     except AttributeError:
    #         return

    def reload(self) -> None:
        self.shm = shared_memory.SharedMemory(name=self._name)
        logger.info(f'Shared memory {self._name} is reloaded at pid {os.getpid()}.')

    def close(self) -> None:
        try:
            del self._data
            self.shm.close()
            logger.info(f'Shared memory {self._name} is closed at pid {os.getpid()}.')
        except AttributeError:
            return

    def __getstate__(self) -> dict:
        return {
            'shape': self.shape,
            'dtype': self.dtype,
            '_name': self._name,
            'shm': self.shm,
        }
