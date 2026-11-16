"""The ``.alignair`` model file: one self-contained file per model.

Sections are either *portable* (``config`` and ``reference_json``: JSON; ``weights``: safetensors; ``reference``:
FASTA; safe and language-agnostic) or *trusted Python-only* (``dataconfig/<i>`` and ``train_state``: pickle, which
executes code on load; only open model files you trust). A model loads its reference from ``reference_json``. Saved
with ``include_trusted_pickle=True`` (the default), a file holds the pickled sections, and with an optimizer it is a
resumable training checkpoint (large: the AdamW moments are about twice the model); ``include_trusted_pickle=False``
writes a publish artifact, the portable sections only. ``io`` saves and loads them, ``container`` is the byte layout and
``serialize`` each section's encoding.
"""
from . import container
from .io import (
    LoadedModel,
    ModelStateData,
    TrainingState,
    TrainingStateData,
    embedded_allele_counts,
    load_model,
    load_training_state,
    pickle_sections,
    read_dataconfig,
    read_metadata,
    read_model_state,
    read_reference,
    read_training_state,
    reference_identity,
    save_model,
    update_card,
)

__all__ = [
    "LoadedModel",
    "ModelStateData",
    "TrainingState",
    "TrainingStateData",
    "container",
    "embedded_allele_counts",
    "load_model",
    "load_training_state",
    "pickle_sections",
    "read_dataconfig",
    "read_metadata",
    "read_model_state",
    "read_reference",
    "read_training_state",
    "reference_identity",
    "save_model",
    "update_card",
]
