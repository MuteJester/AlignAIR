"""What a diagnosed model is, by content: its weights digest, the AlignAIR code that runs it, and its model key.

``weights_sha256`` identifies the weights (a model file's bytes, or an in-memory model's weights, configuration,
module tree and allele order), ``alignair_code`` the ``alignair`` package this process imported (with the torch it
runs on), and ``model_key`` joins the two into the directory name ``workflow.diagnose`` writes a model's diagnostics
under, so a new checkpoint or an AlignAIR code edit never meets another model's results.

The code digest is taken once, when this module is imported: it describes the code the process runs, whatever
happens to the files later. The modules the replay's stages import lazily
(``load_model``'s ``alignair.model_file`` and ``align_germline``'s matcher) are imported here first, so every module
the evaluator runs is loaded before its files are digested.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path

import alignair
import alignair.model_file  # noqa: F401 (loaded before the code digest, see the module docstring)
import alignair.predict.heuristic_matcher  # noqa: F401 (loaded before the code digest, see the module docstring)
import torch

from alignair_assay.core.provenance import code_identity, sha256_file
from alignair_assay.diagnostics.replay import unwrap_model

# The alignair package this process imported.
ALIGNAIR_DIR = Path(alignair.__file__).resolve().parent
# The model key holds this many hex digits of the weights digest and of the code digest.
KEY_DIGITS = 12

_ALIGNAIR_CODE = code_identity(ALIGNAIR_DIR)


def alignair_code() -> dict:
    """``core.provenance.code_identity`` of the alignair package this process imported (``ALIGNAIR_DIR``):
    ``code_sha256`` and ``torch_version``, taken when this module was imported, so it describes the code the
    process runs whatever happens to the files later. When the import is this checkout's ``src/alignair``, it is
    the alignair-local runner's ``local_code_identity()``."""
    return dict(_ALIGNAIR_CODE)


def model_key(weights: str, code_sha256: str) -> str:
    """``<weights>-<code>``: the first ``KEY_DIGITS`` hex digits of the weights digest and of the code digest."""
    return f"{weights[:KEY_DIGITS]}-{code_sha256[:KEY_DIGITS]}"


def weights_sha256(model_or_checkpoint) -> str:
    """The model's identity by content: the SHA-256 of a model file's bytes, or for an in-memory
    ``(model, reference_set)`` pair the SHA-256 of the model's class and module tree, its configuration, the allele
    names of each gene in head order and every state tensor's name, dtype, shape and bytes. A model wrapped by
    ``torch.compile``, ``DataParallel`` or ``DistributedDataParallel`` is the model it wraps
    (``replay.unwrap_model``). FileNotFoundError for a path that is not a file."""
    if isinstance(model_or_checkpoint, (str, os.PathLike)):
        path = Path(model_or_checkpoint).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"no AlignAIR model file {path}")
        return sha256_file(path)
    model, reference_set = model_or_checkpoint
    model = unwrap_model(model)
    digest = hashlib.sha256()
    digest.update(f"{type(model).__module__}.{type(model).__qualname__}\0{model!r}\0".encode())
    cfg = model.cfg
    fields = dataclasses.asdict(cfg) if dataclasses.is_dataclass(cfg) else vars(cfg)
    digest.update(json.dumps(fields, sort_keys=True, default=repr).encode())
    for letter in ("V", "D", "J"):
        if letter != "D" or getattr(reference_set, "has_d", True):
            digest.update(("\0" + letter + "\0" + "\0".join(reference_set.gene(letter).names)).encode())
    for name, value in sorted(model.state_dict().items()):
        if not isinstance(value, torch.Tensor):
            digest.update(f"\0{name}\0{value!r}".encode())
            continue
        tensor = value.detach().to("cpu").contiguous()
        digest.update(f"\0{name}\0{tensor.dtype}\0{tuple(tensor.shape)}\0".encode())
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()
