class ColumnSet:
    """
    A schema for dataset columns that behaves like a list.
    """

    def __init__(self, has_d: bool = False):
        """Initializes the ColumnSet based on whether it has a D segment."""
        # --- Standard attributes ---
        self.sequence: str = 'sequence'
        self.v_call: str = 'v_call'
        self.j_call: str = 'j_call'
        self.productive: str = 'productive'
        self.mutation_rate: str = 'mutation_rate'
        self.indels: str = 'indels'
        self.v_sequence_start: str = 'v_sequence_start'
        self.v_sequence_end: str = 'v_sequence_end'
        self.j_sequence_start: str = 'j_sequence_start'
        self.j_sequence_end: str = 'j_sequence_end'

        # --- Conditional D-segment attributes ---
        if has_d:
            self.d_call: str = 'd_call'
            self.d_sequence_start: str = 'd_sequence_start'
            self.d_sequence_end: str = 'd_sequence_end'
        else:
            self.d_call: str = None
            self.d_sequence_start: str = None
            self.d_sequence_end: str = None

        # --- Cache the list representation ---
        self._list_view = [v for v in vars(self).values() if v is not None and not v.startswith('_')]

    def as_list(self) -> list[str]:
        return self._list_view

    # --- Dunder Methods for List-like Behavior ---

    def __len__(self) -> int:
        return len(self.as_list())

    def __iter__(self):
        return iter(self.as_list())

    def __getitem__(self, index):
        return self.as_list()[index]

    def __contains__(self, item) -> bool:
        return item in self.as_list()

    def __repr__(self) -> str:
        return repr(self.as_list())