"""The .alignair self-contained model format — save/load, selective reads, and the model card."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

import GenAIRR.data as gd

from ..core import AlignAIR
from ..core.config import AlignAIRConfig
from ..reference.reference_set import ReferenceSet
from . import container, serialize

__all__ = ["save_model", "read_metadata", "load_model", "load_training_state",
           "read_dataconfig", "read_reference", "LoadedModel", "TrainingState", "container"]

_DEFAULT_CODECS = {"config": "zlib", "weights": "none", "logvars": "none",
                   "reference": "zlib", "train_state": "zstd", "dataconfig": "zstd"}


def _enum_str(x):
    """Readable string for a GenAIRR enum (Species/ChainType) or plain value; None stays None."""
    if x is None:
        return None
    return str(getattr(x, "value", x))


def _resolve_dataconfigs(dataconfigs):
    """Accept names or DataConfig objects -> list of (name, DataConfig)."""
    out = []
    for dc in dataconfigs:
        if isinstance(dc, str):
            out.append((dc, getattr(gd, dc)))
        else:
            name = getattr(getattr(dc, "metadata", None), "reference_set", None) or type(dc).__name__
            out.append((str(name), dc))
    return out


def _versions():
    import torch
    import GenAIRR
    try:
        from importlib.metadata import version
        av = version("AlignAIR")
    except Exception:
        av = "0+unknown"
    return {"alignair_version": av, "genairr_version": getattr(GenAIRR, "__version__", "?"),
            "torch_version": torch.__version__}


def _provenance():
    import getpass
    import platform
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    return {"git_commit": commit, "host": platform.node(), "user": user}


def save_model(path, model, *, dataconfigs, training, inference=None, logvars=None,
               optimizer=None, rng=None, description="") -> None:
    cfg = model.cfg
    resolved = _resolve_dataconfigs(dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*[dc for _, dc in resolved])

    sections, formats = {}, {}

    def add(name, payload, fmt):
        sections[name] = (payload, _DEFAULT_CODECS.get(name.split("/")[0], "zlib"))
        formats[name] = fmt

    add("config", serialize.config_to_bytes(cfg), "json")
    add("weights", serialize.state_dict_to_bytes(model.state_dict()), "safetensors")
    if logvars is not None:
        add("logvars", serialize.state_dict_to_bytes(logvars.state_dict()), "safetensors")
    add("reference", serialize.reference_fasta(reference).encode("utf-8"), "fasta")
    for i, (_, dc) in enumerate(resolved):
        add(f"dataconfig/{i}", serialize.dataconfig_to_bytes(dc), "python-pickle")
    if optimizer is not None or rng is not None:
        state = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                 "rng": rng or {}, "step": int(training.get("steps", 0)),
                 "train_args": training.get("train_args", {})}
        add("train_state", serialize.train_state_to_bytes(state), "python-pickle")

    bs = int(training.get("batch_size", 0) or 0)
    steps = int(training.get("steps", 0) or 0)
    training = dict(training)
    training.setdefault("total_sequences_seen", steps * bs)
    inf = {"threshold": 0.5, "selector": "absolute", "cap": 3, "germline_reader": "heuristic",
           "pad_mode": "right", "airr": True,
           "chain_types": [ct for ct, _ in resolved] if cfg.num_chain_types > 1 else None}
    if inference:
        inf.update(inference)

    header = {
        "format_version": container.MAJOR_VERSION, "model_class": "AlignAIR", "config_schema_version": 1,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "description": description, "license": "GPL-3.0-or-later", "citation": "AlignAIR",
        **_versions(),
        "model": {"embed_dim": cfg.embed_dim, "max_seq_length": cfg.max_seq_length,
                  "num_chain_types": cfg.num_chain_types, "has_d": cfg.has_d,
                  "param_count": sum(p.numel() for p in model.parameters()),
                  "allele_counts": {"v": cfg.v_allele_count, "d": cfg.d_allele_count,
                                    "j": cfg.j_allele_count}},
        "inference": inf,
        "training": training,
        "reference": {"dataconfigs": [
            {"index": i, "section": f"dataconfig/{i}", "name": name,
             "chain_type": _enum_str(getattr(dc.metadata, "chain_type", None)),
             "species": _enum_str(getattr(dc.metadata, "species", None)),
             "schema_sha256": getattr(dc, "schema_sha256", None)}
            for i, (name, dc) in enumerate(resolved)]},
        "provenance": _provenance(),
        "_formats": formats,
    }
    container.write_container(path, header, sections)


def read_metadata(path) -> dict:
    md = container.read_header(path)
    md.pop("_sections_base", None)
    return md


@dataclass
class LoadedModel:
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    metadata: dict


def _rebuild(path, device):
    md = container.read_header(path)
    cfg = serialize.config_from_bytes(container.read_section(path, "config"))
    model = AlignAIR(cfg).to(device).eval()
    model.load_state_dict(serialize.state_dict_from_bytes(container.read_section(path, "weights")), strict=True)
    n = len(md["reference"]["dataconfigs"])
    dcs = [serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{i}")) for i in range(n)]
    reference = ReferenceSet.from_dataconfigs(*dcs)
    md.pop("_sections_base", None)
    return md, cfg, model, reference


def load_model(path, *, device="cpu") -> LoadedModel:
    md, cfg, model, reference = _rebuild(path, device)
    return LoadedModel(model=model, reference=reference, config=cfg, metadata=md)


@dataclass
class TrainingState:
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    logvars_state: dict | None
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    metadata: dict


def load_training_state(path, *, device="cpu") -> TrainingState:
    md, cfg, model, reference = _rebuild(path, device)
    if "train_state" not in md["sections"]:
        raise ValueError("no train_state section; this model file cannot resume training")
    st = serialize.train_state_from_bytes(container.read_section(path, "train_state"))
    logvars_state = (serialize.state_dict_from_bytes(container.read_section(path, "logvars"))
                     if "logvars" in md["sections"] else None)
    return TrainingState(model=model, reference=reference, config=cfg, logvars_state=logvars_state,
                         optimizer_state=st.get("optimizer"), step=int(st.get("step", 0)),
                         rng=st.get("rng", {}), train_args=st.get("train_args", {}), metadata=md)


def read_dataconfig(path, index=None):
    md = container.read_header(path)
    n = len(md["reference"]["dataconfigs"])
    if index is not None:
        return serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{index}"))
    dcs = [serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{i}")) for i in range(n)]
    return dcs[0] if n == 1 else dcs


def read_reference(path) -> str:
    return container.read_section(path, "reference").decode("utf-8")
