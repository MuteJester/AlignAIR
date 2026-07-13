"""Stable object API (P0-9): the public surface for loading a model and predicting, plus a typed
training entry. Everything here is a thin, documented wrapper over the internal functions
(``load_model`` / ``predict`` / ``train``), so behavior is identical while the surface is stable, typed,
and registry-aware.

    from alignair import Aligner
    aligner = Aligner.from_pretrained("alignair-igh-v1", device="auto")   # path, catalog id, or id@rev
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

    from alignair import TrainingConfig, train
    run = train(TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"), output_dir="runs/igh")
    aligner = run.best_aligner()
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Iterator, Optional, Sequence

__all__ = ["Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "train", "resolve_device"]


def resolve_device(device: str | None = "auto") -> str:
    """Resolve a device request to a concrete backend. ``"auto"`` picks CUDA, else Apple MPS, else CPU.
    An explicit ``"cuda"``/``"mps"`` that is unavailable falls back to CPU (reported by the caller)."""
    import torch
    if device in (None, "auto"):
        if torch.cuda.is_available():
            return "cuda"
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return "mps"
        return "cpu"
    d = str(device)
    if d.startswith("cuda") and not torch.cuda.is_available():
        return "cpu"
    if d.startswith("mps"):
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            return "cpu"
    return d


def _auto_batch_size(device: str) -> int:
    return 256 if device.startswith("cuda") else 64


@dataclass
class PredictionResult:
    """A typed prediction result: the AIRR-convention records plus convenience I/O. Iterable and
    ``len()``-able over the per-read records; ``write_airr`` emits the TSV; ``to_dicts`` returns the raw
    list. The coordinates in each record refer to that record's canonical ``sequence``."""
    records: list
    locus: str = "IGH"
    ids: Optional[list] = None

    def __iter__(self) -> Iterator[dict]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def to_dicts(self) -> list:
        return list(self.records)

    def _ids(self) -> list:
        if self.ids is not None:
            return list(self.ids)
        return [r.get("sequence_id") or f"seq{i}" for i, r in enumerate(self.records)]

    def write_airr(self, path: str, *, columns=None) -> str:
        """Write the records as an AIRR rearrangement TSV. Returns ``path``."""
        from .io.airr import write_airr as _write
        seqs = [r.get("input_sequence", r.get("sequence", "")) for r in self.records]
        _write(path, self._ids(), seqs, self.records, locus=self.locus, columns=columns)
        return path


class Aligner:
    """A loaded model plus its verified germline reference, ready to predict. Construct with
    :meth:`from_pretrained` (a local path, a shipped catalog id, or ``id@revision``) — the single
    inference entry point the CLI is a client of."""

    def __init__(self, model, reference, *, device: str = "cpu", model_path: Optional[str] = None):
        self.model = model
        self.reference = reference
        self.device = device
        self.model_path = model_path

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_pretrained(cls, model: str, *, revision: str | None = None, device: str = "auto",
                        reference=None, dataconfigs=None, registry=None, offline: bool = False,
                        trust_pickle: bool = False) -> "Aligner":
        """Load a model by local path, shipped catalog id, or ``id@revision`` (``revision`` is an alias
        for the ``@`` suffix). Registry ids are resolved + hash-verified into the local cache. ``device``
        accepts ``"auto"`` (CUDA→MPS→CPU) or an explicit backend."""
        from .api import load_model
        from .registry import resolve_model, sources as _sources
        spec = f"{model}@{revision}" if revision and "@" not in str(model) else str(model)
        srcs = _sources.resolve_sources(list(registry) if registry else None)
        path = str(resolve_model(spec, sources=srcs, offline=offline))
        dev = resolve_device(device)
        m, ref = load_model(path, dataconfigs=dataconfigs, reference=reference, device=dev,
                            trust_pickle=trust_pickle)
        return cls(m, ref, device=dev, model_path=path)

    @classmethod
    def from_model(cls, model, reference, *, device: str = "auto") -> "Aligner":
        """Wrap an already-loaded ``(model, reference)`` pair (e.g. straight from training)."""
        dev = resolve_device(device)
        return cls(model.to(dev), reference, device=dev)

    # ------------------------------------------------------------------ properties
    @property
    def loci(self) -> tuple:
        return self.reference.locus_names() if hasattr(self.reference, "locus_names") else ()

    @property
    def default_locus(self) -> str:
        loci = self.loci
        return loci[0] if len(loci) == 1 else "IGH"

    # ------------------------------------------------------------------ inference
    def predict(self, sequences: Sequence[str], *, batch_size="auto", genotype=None,
                genotype_method: str = "mask", airr: bool = True, **overrides) -> PredictionResult:
        """Eagerly align a list of nucleotide strings, returning a :class:`PredictionResult`.
        ``batch_size="auto"`` picks a device-appropriate size. ``genotype`` (a ``{gene: names}`` dict)
        constrains calls to a donor subset of the trained reference."""
        from .api import predict_sequences
        bs = _auto_batch_size(self.device) if batch_size in (None, "auto") else int(batch_size)
        if genotype is not None:
            overrides["genotype"] = genotype
            overrides["genotype_method"] = genotype_method
        records = predict_sequences(self.model, self.reference, list(sequences), device=self.device,
                                    batch_size=bs, airr=airr, **overrides)
        return PredictionResult(records, locus=self.default_locus)

    def predict_iter(self, sequences: "Iterator[str]", *, batch_size="auto", chunk_size: int = 20000,
                     **kw) -> Iterator[dict]:
        """Stream prediction over an iterable of sequences in bounded memory: yields per-read records,
        predicting ``chunk_size`` reads at a time. Same options as :meth:`predict`."""
        buf: list = []
        for s in sequences:
            buf.append(s)
            if len(buf) >= chunk_size:
                yield from self.predict(buf, batch_size=batch_size, **kw).records
                buf = []
        if buf:
            yield from self.predict(buf, batch_size=batch_size, **kw).records


# ============================================================================ training

_PRESETS = {
    "quick": {"steps": 300, "batch_size": 16, "val_every": 100},        # smoke / CI
    "desktop": {"steps": 50_000, "batch_size": 64, "val_every": 2000},  # a single workstation GPU
    "full": {"steps": 300_000, "batch_size": 128, "val_every": 5000},   # a full production run
}


@dataclass(frozen=True)
class TrainingConfig:
    """Typed, immutable training request (replaces unrestricted ``**overrides``). Build with
    :meth:`from_genairr`; ``preset`` supplies resource-tuned defaults you can still override."""
    dataconfigs: tuple
    steps: int = 100_000
    batch_size: int = 64
    lr: float = 3e-4
    device: str = "auto"
    short_boost: int = 1
    grad_clip: Optional[float] = None
    val_every: int = 0
    seed: int = 0

    @classmethod
    def from_genairr(cls, *dataconfigs: str, preset: str | None = None, **overrides) -> "TrainingConfig":
        if preset is not None and preset not in _PRESETS:
            raise ValueError(f"unknown preset {preset!r}; choose from {sorted(_PRESETS)}")
        base = dict(_PRESETS.get(preset, {}))
        base.update(overrides)
        return cls(dataconfigs=tuple(dataconfigs), **base)

    def with_(self, **changes) -> "TrainingConfig":
        return replace(self, **changes)


@dataclass
class TrainingRun:
    """The result of a training run: the final and best checkpoint paths."""
    model_path: str
    best_model_path: Optional[str] = None
    steps: int = 0

    def best_aligner(self, *, device: str = "auto") -> Aligner:
        """Load the best (or final) checkpoint as an :class:`Aligner` ready to predict."""
        return Aligner.from_pretrained(self.best_model_path or self.model_path, device=device)


def train(config: TrainingConfig, *, output_dir: str | None = None, out_path: str | None = None) -> TrainingRun:
    """Train a model from a typed :class:`TrainingConfig`. Writes ``<output_dir>/model.alignair`` (or
    ``out_path``) plus a ``.best.alignair`` when ``val_every > 0``. Returns a :class:`TrainingRun`."""
    import os

    import GenAIRR.data as gd
    from .core import AlignAIR
    from .core.config import AlignAIRConfig
    from .core.losses import make_logvars
    from .reference.reference_set import ReferenceSet
    from .train.guards import validate_training_request
    from .train.trainer import train as _train

    if not (output_dir or out_path):
        raise ValueError("pass output_dir=... or out_path=...")
    path = out_path or os.path.join(output_dir, "model.alignair")
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    dcs = [getattr(gd, d) if isinstance(d, str) else d for d in config.dataconfigs]
    reference = ReferenceSet.from_dataconfigs(*dcs)
    cfg = AlignAIRConfig.from_dataconfigs(*dcs)
    validate_training_request(steps=config.steps, batch_size=config.batch_size, lr=config.lr,
                              max_seq_length=cfg.max_seq_length, reference=reference,
                              grad_clip=config.grad_clip)
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    dev = resolve_device(config.device)
    _train(model, reference, dcs, cfg, logvars, steps=config.steps, batch_size=config.batch_size,
           lr=config.lr, short_boost=config.short_boost, seed=config.seed, device=dev,
           save_path=path, grad_clip=config.grad_clip, val_every=config.val_every)
    stem, ext = os.path.splitext(path)
    best = f"{stem}.best{ext}"
    return TrainingRun(model_path=path, best_model_path=best if os.path.exists(best) else None,
                       steps=config.steps)
