"""Required/optional column schema for AlignAIR CSV input."""
from dataclasses import dataclass


@dataclass
class ColumnSet:
    has_d: bool = True

    @property
    def required_columns(self) -> list[str]:
        cols = ["sequence", "v_call", "j_call",
                "v_sequence_start", "v_sequence_end",
                "j_sequence_start", "j_sequence_end", "mutation_rate"]
        if self.has_d:
            cols += ["d_call", "d_sequence_start", "d_sequence_end"]
        return cols

    @property
    def optional_columns(self) -> list[str]:
        # Defaulted when absent: productive -> 1.0, indels -> count 0.
        return ["productive", "indels"]

    def as_list(self) -> list[str]:
        return self.required_columns + self.optional_columns

    def __iter__(self):
        return iter(self.as_list())

    def __contains__(self, item) -> bool:
        return item in self.as_list()
