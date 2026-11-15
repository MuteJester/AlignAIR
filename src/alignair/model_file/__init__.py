"""The .alignair self-contained model format.

Sections are either *portable* (``config``/json, ``weights``/safetensors, ``reference``/fasta — safe,
language-agnostic) or *trusted Python-only* (``dataconfig``/``train_state`` — pickle, which executes
code on load; only open model files you trust). A file written with an optimizer is a resumable
training checkpoint (large: AdamW moments ~= 2x the model); saving without ``optimizer`` yields a
compact inference-only model. ``io`` saves and loads them, ``container`` is the file layout and ``serialize`` each
section's encoding.
"""
from . import container
from .io import (LoadedModel, ModelStateData, TrainingState, TrainingStateData, embedded_allele_counts,
                 load_model, load_training_state, pickle_sections, read_dataconfig, read_metadata, read_model_state,
                 read_reference, read_training_state, reference_identity, save_model, update_card)

__all__ = ["save_model", "read_metadata", "update_card", "load_model", "load_training_state", "read_training_state",
           "read_model_state", "read_dataconfig", "read_reference", "reference_identity", "embedded_allele_counts",
           "pickle_sections", "LoadedModel", "ModelStateData", "TrainingState", "TrainingStateData", "container"]
