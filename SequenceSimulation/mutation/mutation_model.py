from abc import ABC, abstractmethod


class MutationModel(ABC):
    @abstractmethod
    def apply_mutation(self, sequence):
        pass
