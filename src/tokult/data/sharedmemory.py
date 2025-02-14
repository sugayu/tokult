'''Shared Memory provides us with variables sharing the memory over processes.

This function would be helpful to reduce a memory size in multi-process fitting.
'''

import os
from multiprocessing import shared_memory
import numpy as np
from astropy.nddata import NDData, NDUncertainty
from logging import getLogger

logger = getLogger(__name__)


##
class SharedMemoryNDData(NDData):
    '''SharedMemory wrapper of NDData.'''

    def __init__(self, *args) -> None:
        super().__init__(*args)

    def close(self) -> None:
        self._data_shm.close()
        if self._mask_shm is not None:
            self._mask_shm.unlink()
        self._uncertainty_shm.close()

    def unlink(self) -> None:
        self._data_shm.unlink()
        if self._mask_shm is not None:
            self._mask_shm.unlink()
        self._uncertainty_shm.unlink()

    @property
    def _data(self) -> np.ndarray:
        return self._data_shm.data

    @_data.setter
    def _data(self, data: np.ndarray) -> None:
        self._data_shm = SharedNDArray(data)

    @property
    def _mask(self) -> np.ndarray | None:
        if self._mask_shm is None:
            return None
        return self._mask_shm.data

    @_mask.setter
    def _mask(self, data: np.ndarray | None) -> None:
        if data is None:
            self._mask_shm = None
        else:
            self._mask_shm = SharedNDArray(data)

    @property
    def _uncertainty(self) -> NDUncertainty:
        self._uncertainty_tmp.array = self._uncertainty_shm.data
        return self._uncertainty_tmp

    @_uncertainty.setter
    def _uncertainty(self, uncertainty: NDUncertainty) -> None:
        self._uncertainty_shm = SharedNDArray(uncertainty.array)
        uncertainty.array = np.array([])
        self._uncertainty_tmp: NDUncertainty = uncertainty


class SharedNDArray:
    '''Shared numpy.ndarray.'''

    def __init__(self, data: np.ndarray) -> None:
        shm = shared_memory.SharedMemory(create=True, size=data.nbytes)
        logger.info(f'Shared memory {shm.name} is created.')

        data_tmp: np.ndarray = np.ndarray(data.shape, data.dtype, buffer=shm.buf)
        data_tmp[:] = data[:]
        self.shape = data.shape
        self.dtype = data.dtype
        self.nbytes = data.nbytes
        self._name = shm.name
        self._link_live = True

    @property
    def data(self) -> np.ndarray:
        self.shm: shared_memory.SharedMemory
        try:
            return np.ndarray(self.shape, self.dtype, buffer=self.shm.buf)
        except AttributeError:
            self.shm = shared_memory.SharedMemory(name=self._name)
            logger.info(f'Shared memory {self._name} is loaded at pid {os.getpid()}.')
            return np.ndarray(self.shape, self.dtype, buffer=self.shm.buf)

    def unlink(self) -> None:
        if self._link_live is False:
            logger.debug(f'Shared memory {self._name} is already unlinked.')
            return

        self.shm = shared_memory.SharedMemory(name=self._name)
        try:
            self.shm.close()
            self.shm.unlink()
            logger.info(f'Shared memory {self._name} is unlinked.')
            self._link_live = False
        except AttributeError:
            return

    def close(self) -> None:
        if self._link_live is False:
            logger.debug(f'Shared memory {self._name} is already unlinked.')
            return

        try:
            self.shm.close()
            logger.info(f'Shared memory {self._name} is closed at pid {os.getpid()}.')
            del self.shm
        except AttributeError:
            return
