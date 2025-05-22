from abc import ABC, abstractmethod

class BatchReader(ABC):
    @abstractmethod
    def get_batch(self, pointer: int) -> dict:
        pass

    @abstractmethod
    def get_data_length(self) -> int:
        pass
