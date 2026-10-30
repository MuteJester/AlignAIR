"""Stable object API: the public surface for loading a model and predicting, plus a typed
training entry. Everything here is a thin, documented wrapper over the internal functions
(``load_model`` / ``predict`` / ``train``), so behavior is identical while the surface is stable, typed,
and registry-aware.

    from alignair import Aligner
    aligner = Aligner.from_pretrained("alignair-igh-v1", device="auto")   # path, catalog id, or id@rev
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

    from alignair import TrainingConfig, run_training
    run = run_training(TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", preset="desktop"), output_dir="runs/igh")
    aligner = run.best_aligner()

(The typed training entry is ``run_training`` rather than ``train`` because ``alignair.train`` is the
training subpackage — a top-level ``train`` name would collide with it.)
"""
from __future__ import annotations

from .errors import ConfigurationError, InputError, MissingDependencyError, ModelError

from dataclasses import dataclass, field
from typing import Iterator, Optional, Sequence

__all__ = ["Aligner", "PredictionResult", "TrainingConfig", "TrainingRun", "run_training",
           "resolve_device"]


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
        # the record-owned post-crop pre-orientation read (falls back to canonical); the writer prefers
        # each record's own `input_sequence`, so this equals the CLI's orientation output exactly.
        seqs = [r.get("input_sequence") or r.get("sequence", "") for r in self.records]
        _write(path, self._ids(), seqs, self.records, locus=self.locus, columns=columns)
        return path


class Aligner:
    """A loaded model plus its verified germline reference, ready to predict. Construct with
    :meth:`from_pretrained` (a local path, a shipped catalog id, or ``id@revision``) — the single
    inference entry point the CLI is a client of."""

    def __init__(self, model, reference, *, device: str = "cpu", model_path: Optional[str] = None,
                 source_commit: Optional[str] = None, label_schema=None):
        self.model = model
        self.reference = reference
        self.device = device
        self.model_path = model_path
        self.label_schema = label_schema
        self.source_commit = source_commit     # resolved HF commit SHA (provenance), else None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_pretrained(cls, model: str, *, revision: str | None = None, device: str = "auto",
                        reference=None, dataconfigs=None, registry=None, offline: bool = False,
                        trust_pickle: bool = False, token: str | None = None) -> "Aligner":
        """Load a model by local path, a Hugging Face repo (``hf://org/repo`` or ``org/repo`` — pulled
        via ``huggingface_hub``), or a shipped catalog id (``id`` / ``id@version``). ``revision`` pins a
        HF branch/tag/commit (or catalog version); ``token`` authenticates a private/gated HF repo (or
        ``HF_TOKEN``). ``offline`` uses only local caches. ``device`` accepts ``"auto"`` (CUDA→MPS→CPU)."""
        from .api import load_model
        from .registry import hf, resolve_model
        path = str(resolve_model(model, sources=list(registry) if registry is not None else None,
                                 offline=offline, token=token, revision=revision))
        dev = resolve_device(device)
        m, ref = load_model(path, dataconfigs=dataconfigs, reference=reference, device=dev,
                            trust_pickle=trust_pickle)
        from .model_file import read_metadata, container
        schema = read_metadata(path).get('model', {}).get('label_schema') if container.is_alignair_file(path) else None
        return cls(m, ref, device=dev, model_path=path, source_commit=hf.resolved_commit(path), label_schema=schema)

    def _assert_distributable(self) -> None:
        """Refuse to distribute a resumable checkpoint: those carry trusted **pickle** sections
        (dataconfig / train_state) and must not be published — export a pickle-free inference artifact
        first."""
        if not self.model_path:
            raise ModelError("needs an aligner loaded from a file (model_path is unset)")
        from .model_file import read_metadata
        secs = read_metadata(self.model_path).get("sections", {})
        pickle_secs = [k for k in secs if str(k).startswith("dataconfig/") or k == "train_state"]
        if pickle_secs:
            raise ModelError(
                f"{self.model_path} is a resumable checkpoint with pickle sections {pickle_secs}; it is "
                f"not distributable. Export a pickle-free inference artifact first "
                f"(`alignair convert <in> <out>` / save with include_trusted_pickle=False).")

    def save_pretrained(self, directory: str, *, filename: str = "model.alignair") -> str:
        """Copy the loaded model artifact into ``directory`` (for local distribution or a later
        ``push_to_hub``). Requires an aligner loaded from a **pickle-free** file. Returns the path."""
        import os
        import shutil
        self._assert_distributable()
        os.makedirs(directory, exist_ok=True)
        dest = os.path.join(directory, filename)
        shutil.copyfile(self.model_path, dest)
        return dest

    def push_to_hub(self, repo_id: str, *, token: str | None = None, private: bool = True,
                    filename: str = "model.alignair", create: bool = True, revision: str | None = None):
        """Maintainer-only: upload this model's ``.alignair`` to a Hugging Face repo. Requires
        ``huggingface_hub`` and a write token (arg or ``$HF_TOKEN``). Returns the upload result."""
        import os
        self._assert_distributable()  # never upload a pickle checkpoint
        try:
            from huggingface_hub import HfApi
        except ImportError as e:                    # pragma: no cover - optional dep
            raise MissingDependencyError("push_to_hub needs 'huggingface_hub' (`pip install alignair[hub]`)") from e
        api = HfApi(token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if create:
            api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        return api.upload_file(path_or_fileobj=self.model_path, path_in_repo=filename,
                               repo_id=repo_id, repo_type="model", revision=revision)

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
        records = predict_sequences(self.model, self.reference, sequences, device=self.device,
                                    batch_size=bs, airr=airr, **overrides)
        return PredictionResult(records, locus=self.default_locus)

    def predict_raw(self, sequences, *, batch_size=64) -> dict:
        """Shared raw head arrays for nucleotide/protein models; label_schema describes column identity.

        This method does not apply product calibration or AIRR assembly. Every output uses model
        semantics, including categorical logits versus multilabel probabilities and model-frame positions.
        """
        from .data import prepare_nucleotide_inputs, normalize_sequence_inputs, SequenceValidationPolicy, SequenceInputAdapter
        from .data.inputs import sequence_batch_iterator
        from .core.boundaries import read_window
        from .data.tokenizer import AAProteinTokenizer, AA_TOKEN_DICT, CenterPaddedTokenizer
        from .predict.forward import collect_prediction_outputs, prediction_batches
        from .core.setup import _input_config
        cfg = self.model.cfg
        _input_config(cfg, cfg.input_space)
        window = read_window(cfg.max_seq_length, cfg.boundary_mode)
        if cfg.input_space == 'protein':
            batch = normalize_sequence_inputs(sequence_batch_iterator(sequences),
                SequenceValidationPolicy(frozenset(AA_TOKEN_DICT), window), allow_records=False)
            values = [record.sequence for record in batch.records]
            if not values:
                raise InputError('cannot predict an empty protein batch')
            tokenizer = AAProteinTokenizer(on_invalid='raise')
        else:
            values, _ = prepare_nucleotide_inputs(sequences, window)
            tokenizer = CenterPaddedTokenizer()
        adapter = SequenceInputAdapter(tokenizer, cfg.max_seq_length, self.device, vocab_size=cfg.vocab_size)
        return collect_prediction_outputs(
            prediction_batches(self.model, values, adapter, batch_size=batch_size))

    def predict_iter(self, sequences: "Iterator[str]", *, batch_size="auto", chunk_size: int = 20000,
                     **kw) -> Iterator[dict]:
        """Stream prediction over an iterable of sequences in bounded memory: yields per-read records,
        predicting ``chunk_size`` reads at a time. Same options as :meth:`predict`."""
        from .data.inputs import sequence_batch_iterator
        buf: list = []
        for s in sequence_batch_iterator(sequences):
            buf.append(s)
            if len(buf) >= chunk_size:
                yield from self.predict(buf, batch_size=batch_size, **kw).records
                buf = []
        if buf:
            yield from self.predict(buf, batch_size=batch_size, **kw).records


# Public training names remain available here for existing imports; their owner is alignair.train.
from .train.config import TrainingConfig
from .train.workflow import TrainingRun, run_training
