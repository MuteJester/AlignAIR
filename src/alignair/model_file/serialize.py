"""(De)serialize the individual .alignair sections. Portable: config(json)/weights(safetensors)/
reference(fasta). Trusted Python-only: dataconfig(pickle)/train_state(torch.save)."""
from __future__ import annotations

import io
import json
import pickle

import torch
from safetensors.torch import load as st_load, save as st_save

from ..core.config import AlignAIRConfig

_GENES = ("V", "D", "J")


def config_to_bytes(cfg: AlignAIRConfig) -> bytes:
    return json.dumps(cfg.__dict__).encode("utf-8")


def config_from_bytes(b: bytes) -> AlignAIRConfig:
    return AlignAIRConfig(**json.loads(b.decode("utf-8")))


def state_dict_to_bytes(sd: dict) -> bytes:
    return st_save({k: v.contiguous().cpu() for k, v in sd.items()})


def state_dict_from_bytes(b: bytes) -> dict:
    return st_load(b)


def dataconfig_to_bytes(dc) -> bytes:
    return pickle.dumps(dc, protocol=5)


def dataconfig_from_bytes(b: bytes):
    return pickle.loads(b)


def train_state_to_bytes(state: dict) -> bytes:
    buf = io.BytesIO()
    torch.save(state, buf)
    return buf.getvalue()


def train_state_from_bytes(b: bytes) -> dict:
    return torch.load(io.BytesIO(b), map_location="cpu", weights_only=False)


def reference_fasta(reference) -> str:
    lines = []
    for g in _GENES:
        if g in reference.genes:
            ref = reference.gene(g)
            for name, seq in zip(ref.names, ref.sequences):
                lines.append(f">{name}\n{str(seq).upper()}")
    return "\n".join(lines) + ("\n" if lines else "")
