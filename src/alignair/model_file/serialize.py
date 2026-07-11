"""(De)serialize the individual .alignair sections. Portable: config(json)/weights(safetensors)/
reference(fasta + reference_json). Trusted Python-only: dataconfig(pickle)/train_state(torch.save)."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import pickle
import warnings

import torch
from safetensors.torch import load as st_load, save as st_save

from ..core.config import AlignAIRConfig

_GENES = ("V", "D", "J")


def config_to_bytes(cfg: AlignAIRConfig) -> bytes:
    return json.dumps(cfg.__dict__).encode("utf-8")


def config_from_bytes(b: bytes) -> AlignAIRConfig:
    raw = json.loads(b.decode("utf-8"))
    known = {f.name for f in dataclasses.fields(AlignAIRConfig)}
    extra = sorted(set(raw) - known)               # tolerate config schema drift (e.g. retired fields)
    if extra:
        warnings.warn(f"ignoring unknown AlignAIRConfig field(s) in checkpoint: {extra}")
    return AlignAIRConfig(**{k: v for k, v in raw.items() if k in known})


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


def reference_to_json(reference) -> bytes:
    """Safe (no-pickle) reference section: ordered names + ungapped seqs + gapped V + anchors."""
    return json.dumps(reference.to_serializable(), separators=(",", ":"),
                      ensure_ascii=False).encode("utf-8")


def reference_from_json(b: bytes):
    from ..reference.reference_set import ReferenceSet
    return ReferenceSet.from_serializable(json.loads(b.decode("utf-8")))


def reference_fasta_sha256(reference) -> str:
    """Reference identity: SHA256 of the canonical V,D,J-ordered LF FASTA (== ``reference_fasta``)."""
    return hashlib.sha256(reference_fasta(reference).encode("utf-8")).hexdigest()


def allele_order_sha256(reference) -> str:
    """Model<->reference alignment guard: SHA256 over the ordered V/D/J name lists. The ``D`` key is
    always present (``[]`` when there is no D) so a light-chain reference is unambiguous."""
    def names(g: str):
        return list(reference.gene(g).names) if g in reference.genes else []
    order = {"V": names("V"), "D": names("D") if reference.has_d else [], "J": names("J")}
    payload = json.dumps(order, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
