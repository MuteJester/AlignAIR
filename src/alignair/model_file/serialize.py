"""(De)serialize the individual .alignair sections. Portable: config(json)/weights(safetensors)/
reference(fasta + reference_json). Trusted Python-only: dataconfig(pickle)/train_state(torch.save)."""
from __future__ import annotations

import dataclasses
import hashlib
import io
import json
import pickle

import torch
from safetensors.torch import load as st_load, save as st_save

from ..core.config import AlignAIRConfig
from ..errors import ModelError

_GENES = ("V", "D", "J")


def config_to_bytes(cfg: AlignAIRConfig) -> bytes:
    return json.dumps(cfg.__dict__).encode("utf-8")


# The architecture of AlignAIR 3's network; a config without an ``architecture`` field is one of its files too.
V3_ARCHITECTURE = "conv-v1"
# The settings of AlignAIR 3's configuration that AlignAIR 4's does not have (conv-v1's towers, heads and objectives).
RETIRED_V3_FIELDS = frozenset({
    "filters", "block_out", "latent_size_factor", "v_allele_latent_size", "d_allele_latent_size",
    "j_allele_latent_size", "cls_scale_multiplier", "cls_dim_min", "cls_dim_max", "cls_filter_cap", "state_head",
    "allele_temperatures", "classification_mode", "use_meta_heads", "short_d_penalty", "productive_unknown",
    "region_head", "chip_inputs", "boundary_mode"})


def _config_fields() -> frozenset:
    return frozenset(field.name for field in dataclasses.fields(AlignAIRConfig))


def _v3_refusal(architecture: str) -> ModelError:
    return ModelError(f"this is an AlignAIR 3 model (architecture {architecture!r}); AlignAIR 4 reads chip models "
                      "only — use alignair<4 to run it", context={"architecture": architecture},
                      hint="pip install 'alignair<4' runs it.")


# Settings AlignAIR 4 models once stored, retired at the value every trained model held. A file that stores one
# describes the model this version builds only at that value (d_presence_head: the model's has_d, or None, which meant
# has_d).
RETIRED_V4_SETTINGS = ("use_orientation", "anchor_heads", "reject_empty_sequences", "d_presence_head")


def _retired_values(raw: dict) -> dict:
    has_d = raw.get("has_d", True)
    return {"use_orientation": {True}, "anchor_heads": {True}, "reject_empty_sequences": {False},
            "d_presence_head": {has_d, None}}


def _without_retired_v4_settings(raw: dict, architecture: str) -> dict:
    """``raw`` without the retired AlignAIR 4 settings it stores at their one value; ModelError naming those it stores
    at another (a model this version does not build)."""
    allowed = _retired_values(raw)
    other = {key: raw[key] for key in RETIRED_V4_SETTINGS if key in raw and raw[key] not in allowed[key]}
    if other:
        raise ModelError(f"this model file was built with settings this AlignAIR no longer builds: {other}",
                         context={"architecture": architecture, "settings": sorted(other)},
                         hint="It is an experiment's model; run it with the AlignAIR version that trained it.")
    return {key: value for key, value in raw.items() if key not in RETIRED_V4_SETTINGS}


def config_from_bytes(b: bytes) -> AlignAIRConfig:
    """The model file's configuration. An AlignAIR 3 file (architecture ``conv-v1``; a file without an architecture
    is one) is refused with the version that reads it; a file of another architecture whose configuration holds
    AlignAIR 3's retired settings, with the conversion that rewrites it; a setting this version does not know, as a
    file of a newer AlignAIR."""
    raw = json.loads(b.decode("utf-8"))
    architecture = raw.get("architecture", V3_ARCHITECTURE)
    if architecture == V3_ARCHITECTURE:
        raise _v3_refusal(architecture)
    raw = _without_retired_v4_settings(raw, architecture)
    unknown = sorted(set(raw) - _config_fields())
    if unknown and set(unknown) <= RETIRED_V3_FIELDS:
        from .convert import LAB_CHIPS
        if architecture in LAB_CHIPS:
            raise ModelError(f"this is a lab chip model file (architecture {architecture!r}) of AlignAIR 3's "
                             "configuration: convert it to chip-v1 first",
                             context={"architecture": architecture, "settings": unknown},
                             hint="alignair convert <lab-file>.alignair <chip-file>.alignair")
        raise ModelError(f"this model file (architecture {architecture!r}) was written before AlignAIR 4: its "
                         f"configuration holds AlignAIR 3's settings {unknown}",
                         context={"architecture": architecture, "settings": unknown},
                         hint="Convert it again from the lab chip file it came from (alignair convert), or run it "
                              "with alignair<4.")
    if unknown:
        raise ModelError(f"the model file's configuration has settings this AlignAIR does not know: {unknown}",
                         context={"architecture": architecture, "settings": unknown},
                         hint="It was written by a newer AlignAIR: upgrade alignair to read it.")
    return AlignAIRConfig(**raw)


def lab_config_from_bytes(b: bytes) -> AlignAIRConfig:
    """A lab chip file's configuration with AlignAIR 3's retired settings left out (``convert.convert_lab_chip``
    rewrites it as chip-v1's)."""
    raw = json.loads(b.decode("utf-8"))
    return AlignAIRConfig(**{key: value for key, value in raw.items() if key in _config_fields()})


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
