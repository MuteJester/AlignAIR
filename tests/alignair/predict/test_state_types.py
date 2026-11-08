"""The prediction state names the type of every field (``typing.get_type_hints``): the reads and their flags, the
typed model outputs, each gene's calls and alignments, the V ties, the typed per-read results and the rows, and the
context's model and reference."""
import typing
from typing import Optional

import pytest
import torch

from alignair.predict.outputs import ModelOutputs
from alignair.predict.results import ReadResult
from alignair.predict.state import GeneCall, GermlineAlignment, PredictionContext, PredictionState
from alignair.reference.reference_set import ReferenceSet

pytestmark = pytest.mark.component

NAMES = {"ReadResult": ReadResult, "ModelOutputs": ModelOutputs, "ReferenceSet": ReferenceSet, "torch": torch}


def test_the_prediction_state_names_the_type_of_every_field():
    hints = typing.get_type_hints(PredictionState, localns=NAMES)
    assert hints["reads"] == hints["canonical"] == Optional[list[str]]
    assert hints["cropped"] == hints["no_v"] == Optional[list[bool]]
    assert hints["orientation_given"] == Optional[tuple[Optional[int], ...]]
    assert hints["batches"] == Optional[list[ModelOutputs]]
    assert hints["calls"] == Optional[dict[str, list[GeneCall]]]
    assert hints["alignments"] == Optional[dict[str, list[Optional[GermlineAlignment]]]]
    assert hints["v_ties"] == Optional[list[tuple[str, ...]]]
    assert hints["results"] == Optional[list[ReadResult]]
    assert hints["rows"] == Optional[list[dict]]


def test_the_context_names_the_model_and_reference_types():
    hints = typing.get_type_hints(PredictionContext, localns=NAMES)
    assert hints["model"] is torch.nn.Module and hints["reference"] is ReferenceSet
