"""The .alignair self-contained model format — save/load, selective reads, and the model card.

Sections are either *portable* (``config``/json, ``weights``/safetensors, ``reference``/fasta — safe,
language-agnostic) or *trusted Python-only* (``dataconfig``/``train_state`` — pickle, which executes
code on load; only open model files you trust). A file written with an optimizer is a resumable
training checkpoint (large: AdamW moments ~= 2x the model); saving without ``optimizer`` yields a
compact inference-only model.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from ..errors import ConfigurationError, DataContractError, ModelError, ModelIntegrityError, error_context

from ..core import AlignAIR, validate_reference_binding
from ..core.config import AlignAIRConfig
from ..reference import annotate
from ..reference.reference_set import ReferenceSet
from . import container, serialize

__all__ = ["save_model", "read_metadata", "update_card", "load_model", "load_training_state", "v3_checkpoint_error",
           "read_training_state", "read_model_state", "read_dataconfig", "read_reference", "LoadedModel", "ModelStateData",
           "TrainingState", "TrainingStateData", "container"]

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
    from ..genairr import genairr_identity
    try:
        from importlib.metadata import version
        av = version("AlignAIR")
    except Exception:
        av = "0+unknown"
    return {"alignair_version": av, **genairr_identity(), "torch_version": torch.__version__}


def _provenance(*, publish: bool = False):
    """The source commit, and for a file that stays with its run (a resumable checkpoint) the machine and login it
    was written on; a file made to be shared (``publish``) names neither."""
    import getpass
    import platform
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    if publish:
        return {"git_commit": commit}
    try:
        user = getpass.getuser()
    except Exception:
        user = None
    return {"git_commit": commit, "host": platform.node(), "user": user}


_RESERVED_CARD_KEYS = {"format_version", "model_format_version", "model_class", "model_id",
                       "model_version", "sections", "_formats", "reference", "model", "training",
                       "inference", "provenance", "created", "created_by_alignair"}


def save_model(path, model, *, dataconfigs, training, inference=None,
               optimizer=None, rng=None, description="", include_trusted_pickle=True,
               model_id=None, model_version=None, card=None, label_schema=None, scheduler=None) -> None:
    """Write an .alignair model file. ``include_trusted_pickle=True`` (default) keeps the pickled
    ``dataconfig``/``train_state`` sections (a resumable training checkpoint). Set it to ``False`` for
    a **publish/inference artifact**: weights + config + safe ``reference_json`` + ``reference`` FASTA
    + model card, ZERO pickle sections, and no host or user in its provenance. ``card`` supplies descriptive AIRR fields (species/locus/
    intended_use/…) merged into the header. A model may record its ordered
    ``label_schema``; allele models may omit it. Binding validation precedes destination writes.
    """
    cfg = model.cfg
    resolved = _resolve_dataconfigs(dataconfigs)
    reference = ReferenceSet.from_dataconfigs(*[dc for _, dc in resolved])
    with error_context(stage='model_binding', path=path):
        validate_reference_binding(reference, cfg, label_schema=label_schema)

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
    inf = {"threshold": 0.5, "selector": "absolute", "cap": 3,
           "pad_mode": "right", "airr": True,
           "chain_types": [ct for ct, _ in resolved] if cfg.num_chain_types > 1 else None}
    if inference:
        inf.update(inference)

    header = {
        "format_version": container.MAJOR_VERSION, "model_format_version": container.MAJOR_VERSION,
        "model_class": "AlignAIR", "config_schema_version": 1,
        "model_id": model_id, "model_version": model_version,
        "created": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "created_by_alignair": _versions()["alignair_version"],
        "description": description, "license": "GPL-3.0-or-later", "citation": "AlignAIR",
        **_versions(),
        "model": {"embed_dim": cfg.embed_dim, "max_seq_length": cfg.max_seq_length,
                  "input_space": cfg.input_space, "label_space": cfg.label_space,
                  "label_schema": label_schema,
                  "num_chain_types": cfg.num_chain_types, "has_d": cfg.has_d,
                  "param_count": sum(p.numel() for p in model.parameters()),
                  "allele_counts": {"v": cfg.v_allele_count, "d": cfg.d_allele_count,
                                    "j": cfg.j_allele_count}},
        "inference": inf,
        "training": training,
        "reference": {
            "reference_fasta_sha256": serialize.reference_fasta_sha256(reference),
            "allele_order_sha256": serialize.allele_order_sha256(reference),
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


def read_metadata(path) -> dict:
    md = container.read_header(path)
    md.pop("_sections_base", None)
    return md


# header fields that are structural / integrity-bearing and must never be edited by update_card:
# changing them would either corrupt the container or desync the card from the embedded model/reference.
_UNEDITABLE_CARD_KEYS = {"format_version", "model_format_version", "model_class", "config_schema_version",
                         "sections", "_formats", "_sections_base", "model", "reference"}


def update_card(src_path, dst_path, updates: dict) -> dict:
    """Maintainer utility: rewrite an .alignair with patched header/card fields (e.g. set
    ``model_id``/``model_version``/``locus``/``species`` on an inference artifact before publishing),
    preserving **every section byte-for-byte** — the payloads (weights, config, reference, reference_json)
    are re-emitted unchanged, so the loaded model and all reference/integrity hashes are identical. Only
    descriptive header fields may change; structural/integrity keys (:data:`_UNEDITABLE_CARD_KEYS`) are
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


# The first bytes of a torch checkpoint (an AlignAIR 3 ``.pt``): a zip archive, or a pickle stream (protocol >= 2).
_TORCH_MAGIC = (b"PK\x03\x04", b"\x80")


def v3_checkpoint_error(path) -> ModelError:
    """The refusal of a path that is not an ``.alignair`` model file: a missing file, an AlignAIR 3 checkpoint (a torch
    ``.pt``, refused with the version that runs it), or a file that is no model at all."""
    from pathlib import Path
    target = Path(path)
    if not target.is_file():
        return ModelError(f"no such model file: {path}", context={"path": str(path)},
                          hint="Check the path, or name a shipped model by its id (alignair models list).")
    with open(target, "rb") as handle:
        head = handle.read(4)
    if head.startswith(_TORCH_MAGIC):
        return ModelError(f"{path} is an AlignAIR 3 checkpoint (not an .alignair model file); AlignAIR 4 reads chip "
                          "models only — use alignair<4 to run it", context={"path": str(path)},
                          hint="pip install 'alignair<4' runs it.")
    return ModelError(f"{path} is not an AlignAIR model file (.alignair)", context={"path": str(path)},
                      hint="An empty or truncated file is an interrupted download: fetch the model again.")


@dataclass
class LoadedModel:
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


def read_model_state(path) -> ModelStateData:
    """Decode weights and verify reference/label bindings without model construction or RNG use.

    The reference comes from the no-pickle ``reference_json``, the weights from safetensors. This does not decode training state, require architecture factories, or
    check architecture-specific weight keys/shapes; restoration owns those checks.
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
    schema = md.get('model', {}).get('label_schema')
    with error_context(stage='model_binding', path=path):
        _verify_reference_integrity(reference, md)
        try:
            validate_reference_binding(reference, cfg, label_schema=schema)
        except (ConfigurationError, DataContractError) as exc:
            message = ('stored label schema disagrees with the model/reference' if schema is not None
                       else 'stored model/reference binding is invalid')
            raise ModelIntegrityError(f'{message}: {exc}',
                                      context={'field': 'label_schema' if schema is not None else 'reference',
                                               **exc.context}, hint=exc.hint) from exc
    model_state = serialize.state_dict_from_bytes(container.read_section(path, "weights"))
    md.pop("_sections_base", None)
    return ModelStateData(md, cfg, model_state, reference)


def load_model(path, *, device="cpu", reference=None) -> LoadedModel:
    """Load an inference model bound to its embedded reference, or to ``reference``: any reference with its genes (the
    model's alleles are an input), annotated against the embedded one (``alignair.reference.annotate``: the model's
    own alleles keep its anchors and gapped forms, a novel one takes them as training would)."""
    data = read_model_state(path)
    model = AlignAIR.from_state_dict(data.config, data.model_state, device=device, source=path)
    bound = data.reference if reference is None else annotate(reference, known=data.reference)
    model.bind_reference(bound)
    return LoadedModel(model=model, reference=bound, config=model.cfg, metadata=data.metadata)


@dataclass
class TrainingState:
    model: AlignAIR
    reference: ReferenceSet
    config: AlignAIRConfig
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    metadata: dict
    scheduler_state: dict | None = None


@dataclass
class TrainingStateData:
    """Decoded checkpoint tensors and validated bindings, without an allocated model instance."""
    model_state: dict
    reference: ReferenceSet
    config: AlignAIRConfig
    optimizer_state: dict | None
    step: int
    rng: dict
    train_args: dict
    metadata: dict
    scheduler_state: dict | None = None


def read_training_state(path, *, device="cpu") -> TrainingStateData:
    """Decode a trusted .alignair training checkpoint without constructing a model.

    Section checksums and reference/label bindings are checked here. Architecture-specific weight
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
        return TrainingStateData(model_state=model_state, reference=reference, config=cfg,
                                 optimizer_state=st.get("optimizer"),
                                 step=int(st.get("step", 0)), rng=st.get("rng", {}),
                                 train_args=st.get("train_args", {}), metadata=md,
                                 scheduler_state=st.get('scheduler'))


def load_training_state(path, *, device="cpu") -> TrainingState:
    """Decode a trusted checkpoint and restore its model, retaining the existing model-bearing API."""
    data = read_training_state(path)
    model = AlignAIR.from_state_dict(data.config, data.model_state, device=device, source=path)
    model.bind_reference(data.reference)
    return TrainingState(model=model, reference=data.reference, config=model.cfg,
                         optimizer_state=data.optimizer_state,
                         step=data.step, rng=data.rng, train_args=data.train_args,
                         metadata=data.metadata, scheduler_state=data.scheduler_state)


def read_dataconfig(path, index=None):
    md = container.read_header(path)
    n = len(md["reference"]["dataconfigs"])
    if index is not None:
        return serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{index}"))
    dcs = [serialize.dataconfig_from_bytes(container.read_section(path, f"dataconfig/{i}")) for i in range(n)]
    return dcs[0] if n == 1 else dcs


def read_reference(path) -> str:
    return container.read_section(path, "reference").decode("utf-8")
