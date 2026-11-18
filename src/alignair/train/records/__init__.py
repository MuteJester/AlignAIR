"""The training records: the curriculum, the record streams (their seed blocks, novel alleles) and the worker
processes that draw them."""
from .curriculum import Curriculum
from .streams import CurriculumStreamPolicy, RecordSource, training_records

__all__ = ["Curriculum", "CurriculumStreamPolicy", "RecordSource",
           "training_records"]
