from enum import Enum, auto

class MutationModels(Enum):
    UNIFORM = auto()
    S5F = auto()

class MutationConfig:
    def __init__(self,mutation_rate,model=MutationModels.UNIFORM):
        self.mutation_rate = mutation_rate
        self.model = model