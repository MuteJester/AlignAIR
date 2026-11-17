"""Saving and loading .alignair model files, their selective reads and the model card.

Sections are either *portable* (``config`` and ``reference_json``: JSON; ``weights``: safetensors; ``reference``:
FASTA; safe and language-agnostic) or *trusted Python-only* (``dataconfig/<i>`` and ``train_state``: pickle, which
executes code on load; only open model files you trust). A model loads its reference from ``reference_json``. Saved
with ``include_trusted_pickle=True`` (the default), a file holds the pickled sections, and with an optimizer it is a
resumable training checkpoint (large: the AdamW moments are about twice the model); ``include_trusted_pickle=False``
writes a publish artifact, the portable sections only.
"""
from __future__ import annotations

import os
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from ..core import AlignAIR, validate_reference_binding
from ..core.config import AlignAIRConfig
from ..errors import ConfigurationError, DataContractError, ModelError, ModelIntegrityError, error_context
from ..reference import annotate
from ..reference.reference_set import ReferenceSet
from . import container, serialize

if TYPE_CHECKING:
    import torch
    from GenAIRR import DataConfig

_DEFAULT_CODECS = {"config": "zlib", "weights": "none",
                   "reference": "zlib", "train_state": "zstd", "dataconfig": "zstd"}


def _enum_str(x):
    """Readable string for a GenAIRR enum (Species/ChainType) or plain value; None stays None."""
    if x is None:
        return None
    return str(getattr(x, "value", x))


def _resolve_dataconfigs(dataconfigs):
    """Accept names or DataConfig objects -> list of (name, DataConfig)."""
    from ..genairr import name_of, resolve_dataconfig
    out = []
    for spec in dataconfigs:
        dc = resolve_dataconfig(spec)
        name = (spec if isinstance(spec, str) else name_of(dc)
                or getattr(getattr(dc, "metadata", None), "reference_set", None) or type(dc).__name__)
        out.append((str(name), dc))
    return out


def _versions():
    import torch

    from .. import __version__
    from ..genairr import genairr_identity
    return {"alignair_version": __version__, **genairr_identity(), "torch_version": torch.__version__}


def _package_dir() -> Path:
    """The directory of the ``alignair`` package that is running."""
    return Path(__file__).resolve().parent.parent


def _source_commit() -> str | None:
    """The commit of the AlignAIR checkout this package runs from; None for an installed package (its files are not
    tracked by any repository), whatever repository the working directory is in."""
    import subprocess
    package = str(_package_dir())
    try:
        subprocess.check_output(["git", "-C", package, "ls-files", "--error-unmatch", "__init__.py"],
                                stderr=subprocess.DEVNULL)
        return subprocess.check_output(["git", "-C", package, "rev-parse", "HEAD"],
                                       stderr=subprocess.DEVNULL).decode().strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _provenance(*, publish: bool = False):
    """The AlignAIR source commit, and for a file that stays with its run (a resumable checkpoint) the machine and
    login it was written on; a file made to be shared (``publish``) names neither."""
    import getpass
    import platform
    commit = _source_commit()
    if publish:
        return {"git_commit": commit}
    try:
        user = getpass.getuser()
    except (OSError, KeyError):
        user = None
    return {"git_commit": commit, "host": platform.node(), "user": user}


# The header's keys by who sets them. save_model writes all of them but a card's: the container's own (structural) and
# the description of the embedded model and reference (integrity) never change after it; the run's record and the
# catalog identity (record) may be restamped by update_card. A card (save_model's ``card``) sets any other key.
_STRUCTURAL_KEYS = frozenset({"format_version", "model_format_version", "model_class", "config_schema_version",
                              "sections", "_formats", "_sections_base"})
_INTEGRITY_KEYS = frozenset({"model", "reference"})
_RECORD_KEYS = frozenset({"model_id", "model_version", "training", "provenance", "created", "created_by_alignair"})
_RESERVED_CARD_KEYS = _STRUCTURAL_KEYS | _INTEGRITY_KEYS | _RECORD_KEYS
_UNEDITABLE_CARD_KEYS = _STRUCTURAL_KEYS | _INTEGRITY_KEYS


def reference_identity(reference: ReferenceSet) -> dict:
    """The identity a model file records of its reference: the digests of its allele order and of its sequences."""
    return {"reference_fasta_sha256": serialize.reference_fasta_sha256(reference),
            "allele_order_sha256": serialize.allele_order_sha256(reference)}


def save_model(path: str | os.PathLike[str], model: AlignAIR, reference: ReferenceSet, *,
               dataconfigs: Iterable[str | DataConfig] = (), training: dict,
               optimizer: torch.optim.Optimizer | None = None, rng: dict | None = None, description: str = "",
               include_trusted_pickle: bool = True,
               model_id: str | None = None, model_version: str | None = None, card: dict | None = None,
               scheduler: torch.optim.lr_scheduler.LRScheduler | None = None) -> None:
    """Write ``model``, bound to ``reference``, as an .alignair model file: its weights, its configuration and
    ``reference`` (``reference_json``, and a FASTA), whose digests identify the model. ``dataconfigs`` (names or GenAIRR
    DataConfigs) are the sources the model was trained from, recorded by name. ``include_trusted_pickle=True`` (the
    default) also stores them pickled, and the optimizer, scheduler and random state, as a resumable training
    checkpoint; ``False`` writes a **publish/inference artifact**: no pickle section, and no host or user in its
    provenance. ``card`` adds descriptive fields (species, locus, intended use...) to the header. The binding is
    checked before anything is written."""
    cfg = model.cfg
    resolved = _resolve_dataconfigs(dataconfigs)
    with error_context(stage='model_binding', path=path):
        validate_reference_binding(reference, cfg)

    sections, formats = {}, {}

    def add(name, payload, fmt):
        sections[name] = (payload, _DEFAULT_CODECS.get(name.split("/")[0], "zlib"))
        formats[name] = fmt

    add("config", serialize.config_to_bytes(cfg), "json")
    add("weights", serialize.state_dict_to_bytes(model.state_dict()), "safetensors")
    add("reference", serialize.reference_fasta(reference).encode("utf-8"), "fasta")
    add("reference_json", serialize.reference_to_json(reference), "json")   # safe inference load source
    if include_trusted_pickle:                                             # only in trusted checkpoints
        for i, (_, dc) in enumerate(resolved):
            add(f"dataconfig/{i}", serialize.dataconfig_to_bytes(dc), "python-pickle")
        if optimizer is not None or rng is not None or scheduler is not None:
            state = {"optimizer": optimizer.state_dict() if optimizer is not None else None,
                     "scheduler": scheduler.state_dict() if scheduler is not None else None,
                     "rng": rng or {}, "step": int(training.get("steps", 0)),
                     "train_args": training.get("train_args", {})}
            add("train_state", serialize.train_state_to_bytes(state), "python-pickle")

    bs = int(training.get("batch_size", 0) or 0)
    steps = int(training.get("steps", 0) or 0)
    training = dict(training)
    training.setdefault("total_sequences_seen", steps * bs)

    versions = _versions()
    header = {
        "format_version": container.MAJOR_VERSION, "model_format_version": container.MAJOR_VERSION,
        "model_class": "AlignAIR", "config_schema_version": 1,
        "model_id": model_id, "model_version": model_version,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by_alignair": versions["alignair_version"],
        "description": description, "license": "GPL-3.0-or-later", "citation": "AlignAIR",
        **versions,
        "model": {"embed_dim": cfg.embed_dim, "max_seq_length": cfg.max_seq_length,
                  "num_chain_types": cfg.num_chain_types, "has_d": cfg.has_d,
                  "param_count": sum(p.numel() for p in model.parameters())},
        "training": training,
        "reference": {
            **reference_identity(reference),
            "dataconfigs": [
                {"index": i, "section": (f"dataconfig/{i}" if include_trusted_pickle else None),
                 "name": name, "chain_type": _enum_str(getattr(dc.metadata, "chain_type", None)),
                 "species": _enum_str(getattr(dc.metadata, "species", None)),
                 "schema_sha256": getattr(dc, "schema_sha256", None)}
                for i, (name, dc) in enumerate(resolved)]},
        "provenance": _provenance(publish=not include_trusted_pickle),
        "_formats": formats,
    }
    for k, v in (card or {}).items():                     # descriptive AIRR fields (species/locus/…)
        if k not in _RESERVED_CARD_KEYS:
            header[k] = v
    container.write_container(path, header, sections)


def read_metadata(path: str | os.PathLike[str]) -> dict:
    """A model file's JSON header: its metadata and section index (``container``)."""
    md = container.read_header(path)
    md.pop("_sections_base", None)
    return md


def pickle_sections(metadata: dict) -> list[str]:
    """The sections of a model file's ``metadata`` (``read_metadata``) that hold pickles: the dataconfigs and the
    training state a resumable checkpoint carries. A file with one is not distributable."""
    return [name for name in metadata.get("sections", {}) if name.startswith("dataconfig/") or name == "train_state"]


def update_card(src_path: str | os.PathLike[str], dst_path: str | os.PathLike[str], updates: dict) -> dict:
    """Maintainer utility: rewrite an .alignair with patched header/card fields (e.g. set
    ``model_id``/``model_version``/``locus``/``species`` on an inference artifact before publishing),
    preserving **every section byte-for-byte** — the payloads (weights, config, reference, reference_json)
    are re-emitted unchanged, so the loaded model and all reference/integrity hashes are identical. Only
    descriptive and record header fields may change; structural and integrity keys (``_UNEDITABLE_CARD_KEYS``) are
    rejected. Returns the new metadata. This edits metadata only — it cannot publish or upload."""
    header = container.read_header(src_path)
    sections, formats = {}, {}
    for name, s in header.get("sections", {}).items():
        sections[name] = (container.read_section(src_path, name), s["codec"])   # verifies checksums
        formats[name] = s.get("format", "bytes")
    new_header = {k: v for k, v in header.items() if not k.startswith("_") and k != "sections"}
    bad = _UNEDITABLE_CARD_KEYS & set(updates)
    if bad:
        raise ValueError(f"update_card cannot change structural/integrity fields: {sorted(bad)}")
    new_header.update(updates)
    new_header["_formats"] = formats
    container.write_container(dst_path, new_header, sections)
    return read_metadata(dst_path)


@dataclass
class LoadedModel:
    """A loaded model file: the model bound to its reference, its configuration and its metadata."""
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    metadata: dict


def _verify_reference_integrity(reference, md) -> None:
    """Reject a reference whose allele ORDER (or content) does not match the model card — a wrong-order
    reference silently misaligns every allele call to the model head."""
    ref_md = md.get("reference", {})
    order = ref_md.get("allele_order_sha256")
    if order and serialize.allele_order_sha256(reference) != order:
        raise ModelIntegrityError("allele_order_sha256 mismatch: the embedded reference order does not match "
                         "the model card; allele calls would be misaligned to the model head.")
    fasta = ref_md.get("reference_fasta_sha256")
    if fasta and serialize.reference_fasta_sha256(reference) != fasta:
        raise ModelIntegrityError("reference_fasta_sha256 mismatch: the embedded reference differs from the card.")


@dataclass
class ModelStateData:
    """Decoded CPU weights and checked bindings, without allocating architecture modules."""
    metadata: dict
    config: AlignAIRConfig
    model_state: dict
    reference: ReferenceSet


def embedded_allele_counts(path: str | os.PathLike[str]) -> dict[str, int]:
    """Per gene (``"v"``, ``"d"``, ``"j"``), how many alleles of its embedded reference a model file's model calls: 0
    for the D of a model without one, whatever its reference holds. A file without ``reference_json`` (an AlignAIR 3
    model) gives the counts its header stores, if any. No weight is read."""
    md = container.read_header(path)
    model = md.get("model") or {}
    if "reference_json" not in md["sections"]:
        return dict(model.get("allele_counts") or {})
    reference = serialize.reference_from_json(container.read_section(path, "reference_json"))
    has_d = model.get("has_d", reference.has_d) and "D" in reference.genes
    return {"v": len(reference.gene("V")), "d": len(reference.gene("D")) if has_d else 0,
            "j": len(reference.gene("J"))}


def read_model_state(path: str | os.PathLike[str]) -> ModelStateData:
    """Decode weights and verify the reference binding without model construction or RNG use.

    The reference comes from the no-pickle ``reference_json``, the weights from safetensors. This does not decode
    training state, require architecture factories, or check architecture-specific weight keys/shapes; restoration
    owns those checks.
    """
    md = container.read_header(path)
    cfg = serialize.config_from_bytes(container.read_section(path, "config"))
    if "reference_json" not in md["sections"]:
        raise ModelError(f"{path} has no reference_json: it was written before AlignAIR 4",
                         context={"path": str(path)}, hint="Run it with alignair<4.")
    try:                                         # a fault in the file's own reference is the file's
        reference = serialize.reference_from_json(container.read_section(path, "reference_json"))
    except (DataContractError, ValueError, KeyError, TypeError) as exc:
        if isinstance(exc, ModelError):
            raise
        raise ModelIntegrityError(f"{path} holds a reference this build cannot read: {exc}",
                                  context={"path": str(path), "field": "reference_json"}) from exc
    with error_context(stage='model_binding', path=path):
        _verify_reference_integrity(reference, md)
        try:
            validate_reference_binding(reference, cfg)
        except (ConfigurationError, DataContractError) as exc:
            raise ModelIntegrityError(f'stored model/reference binding is invalid: {exc}',
                                      context={'field': 'reference', **exc.context}, hint=exc.hint) from exc
    model_state = serialize.state_dict_from_bytes(container.read_section(path, "weights"))
    md.pop("_sections_base", None)
    return ModelStateData(md, cfg, model_state, reference)


def load_model(path: str | os.PathLike[str], *, device: str = "cpu",
               reference: ReferenceSet | None = None) -> LoadedModel:
    """Load an inference model bound to its embedded reference, or to ``reference``: any reference with its genes (the
    model's alleles are an input), annotated against the embedded one (``alignair.reference.annotate``: the model's
    own alleles keep its anchors and gapped forms, a novel one takes them as training would)."""
    data = read_model_state(path)
    model = AlignAIR.from_state_dict(data.config, data.model_state, device=device, source=path)
    bound = data.reference if reference is None else annotate(reference, known=data.reference)
    model.bind_reference(bound)
    return LoadedModel(model=model, reference=bound, config=model.cfg, metadata=data.metadata)


@dataclass(kw_only=True)
class TrainingStateData(ModelStateData):
    """A training checkpoint's model state and the state its training resumes from, without an allocated model."""
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    scheduler_state: dict | None = None


@dataclass(kw_only=True)
class TrainingState(LoadedModel):
    """A training checkpoint's restored model and the state its training resumes from."""
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    scheduler_state: dict | None = None


def read_training_state(path: str | os.PathLike[str], *, device: str = "cpu") -> TrainingStateData:
    """Decode a trusted .alignair training checkpoint without constructing a model.

    Section checksums and the reference binding are checked here. Architecture-specific weight
    keys/shapes are checked when restoring into a model. Model tensors are mapped to ``device``;
    optimizer, scheduler and RNG state retain their serialized values. Like
    ``load_training_state``, this is a trusted resume flow that decodes pickle training sections.
    """
    with error_context(stage="training_checkpoint_load", path=path):
        data = read_model_state(path)
        md, cfg, model_state, reference = data.metadata, data.config, data.model_state, data.reference
        if "train_state" not in md["sections"]:
            raise ModelError("no train_state section; this model file cannot resume training")
        st = serialize.train_state_from_bytes(container.read_section(path, "train_state"))
        model_state = {name: value.to(device) for name, value in model_state.items()}
        return TrainingStateData(metadata=md, config=cfg, model_state=model_state, reference=reference,
                                 optimizer_state=st.get("optimizer"),
                                 step=int(st.get("step", 0)), rng=st.get("rng", {}),
                                 train_args=st.get("train_args", {}), scheduler_state=st.get('scheduler'))


def load_training_state(path: str | os.PathLike[str], *, device: str = "cpu") -> TrainingState:
    """Decode a trusted checkpoint and restore its model, retaining the existing model-bearing API."""
    data = read_training_state(path)
    model = AlignAIR.from_state_dict(data.config, data.model_state, device=device, source=path)
    model.bind_reference(data.reference)
    return TrainingState(model=model, reference=data.reference, config=model.cfg,
                         optimizer_state=data.optimizer_state,
                         step=data.step, rng=data.rng, train_args=data.train_args,
                         metadata=data.metadata, scheduler_state=data.scheduler_state)


def read_dataconfig(path: str | os.PathLike[str], index: int) -> DataConfig:
    """The pickled GenAIRR DataConfig section ``dataconfig/<index>`` of a training checkpoint (trusted input only: it
    is unpickled)."""
    return serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{index}"))


def read_reference(path: str | os.PathLike[str]) -> str:
    """A model file's reference as FASTA text (its ``reference`` section)."""
    return container.read_section(path, "reference").decode("utf-8")


def read_reference_gapped(path: str | os.PathLike[str]) -> str:
    """``read_reference``'s FASTA with each V allele IMGT-gapped (the form Change-O's ``CreateGermlines.py`` reads),
    D and J as they are. A ConfigurationError naming the V alleles the reference holds no gapped form of."""
    v = serialize.reference_from_json(container.read_section(path, "reference_json")).gene("V")
    gapped = v.gapped or {}
    missing = [name for name in v.names if name not in gapped]
    if missing:
        raise ConfigurationError(f"the model's reference has no IMGT-gapped form of {len(missing)} V allele(s): "
                                 f"{', '.join(missing[:5])}", context={"alleles": missing},
                                 hint="Export without --gapped, or gap these alleles against IMGT's.")
    records: list[list[str]] = []
    for line in read_reference(path).splitlines():
        if line.startswith(">"):
            records.append([line, ""])
        elif records:
            records[-1][1] += line.strip()
    return "".join(f"{header}\n{gapped.get(header[1:].split()[0], sequence)}\n" for header, sequence in records)
