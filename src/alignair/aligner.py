"""The inference surface: load a model (``Aligner.from_pretrained``) and predict, a thin, typed and registry-aware
wrapper over ``alignair.api``.

    from alignair import Aligner
    aligner = Aligner.from_pretrained("alignair4-igh-human", device="auto")   # catalog id (PENDING-V4-PUBLISH), path, or id@version
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

Training is ``alignair.run_training`` (``alignair.train``).
"""
from __future__ import annotations

from .errors import MissingDependencyError, ModelError
from .reference.reference_set import FALLBACK_LOCUS

from dataclasses import dataclass
from typing import Iterator, Optional, Sequence

__all__ = ["Aligner", "PredictionResult", "resolve_device"]


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


def _auto_batch_size(device: str, workers: int = 1) -> int:
    """256 on a GPU, 64 in one CPU process, 16 per single-threaded CPU worker (``predict.parallel``), whose
    activations then stay in its share of the cache."""
    if device.startswith("cuda"):
        return 256
    return 16 if workers > 1 else 64


@dataclass
class PredictionResult:
    """A typed prediction result: the AIRR-convention records plus convenience I/O. Iterable and
    ``len()``-able over the per-read records; ``write_airr`` emits the TSV; ``to_dicts`` returns the raw
    list. The coordinates in each record refer to that record's canonical ``sequence``."""
    records: list
    locus: str = FALLBACK_LOCUS
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
                 source_commit: Optional[str] = None):
        self.model = model
        self.reference = reference
        self.device = device
        self.model_path = model_path
        self.source_commit = source_commit     # resolved HF commit SHA (provenance), else None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_pretrained(cls, model: str, *, revision: str | None = None, device: str = "auto",
                        reference=None, registry=None, offline: bool = False,
                        token: str | None = None) -> "Aligner":
        """Load a model by local path, a Hugging Face repo (``hf://org/repo`` or ``org/repo`` — pulled
        via ``huggingface_hub``), or a shipped catalog id (``id`` / ``id@version``). ``revision`` pins a
        HF branch/tag/commit (or catalog version); ``token`` authenticates a private/gated HF repo (or
        ``HF_TOKEN``). ``offline`` uses only local caches. ``device`` accepts ``"auto"`` (CUDA→MPS→CPU)."""
        from .api import load_model
        from .registry import hf, resolve_model
        path = str(resolve_model(model, sources=list(registry) if registry is not None else None,
                                 offline=offline, token=token, revision=revision))
        dev = resolve_device(device)
        m, ref = load_model(path, reference=reference, device=dev)
        return cls(m, ref, device=dev, model_path=path, source_commit=hf.resolved_commit(path))

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
                f"not distributable. Export a pickle-free bundle first: `alignair export {self.model_path} "
                f"<out_dir>` (alignair.train.build.export_bundle), then share <out_dir>/model.alignair.")

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
        """Maintainer-only: upload this model's ``.alignair`` to a Hugging Face repo of its own, in one commit with the
        bundle's ``model_card.md`` and ``reference_manifest.json`` when they sit beside the model file (as
        ``alignair export`` writes them). Requires ``huggingface_hub`` and a write token (arg or ``$HF_TOKEN``). For
        the shared catalog repo, use ``alignair.registry.publish_bundle`` and ``upload_version``. Returns the commit."""
        import os
        self._assert_distributable()  # never upload a pickle checkpoint
        try:
            import huggingface_hub
        except ImportError as e:                    # pragma: no cover - optional dep
            raise MissingDependencyError("push_to_hub needs 'huggingface_hub' (`pip install alignair[hub]`)") from e
        api = huggingface_hub.HfApi(token=token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"))
        if create:
            api.create_repo(repo_id, private=private, exist_ok=True, repo_type="model")
        folder = os.path.dirname(os.path.abspath(self.model_path))
        files = [(self.model_path, filename)] + [(os.path.join(folder, name), name)
                                                 for name in ("model_card.md", "reference_manifest.json")
                                                 if os.path.isfile(os.path.join(folder, name))]
        operations = [huggingface_hub.CommitOperationAdd(path_in_repo=dest, path_or_fileobj=src) for src, dest in files]
        return api.create_commit(repo_id=repo_id, operations=operations, repo_type="model", revision=revision,
                                 commit_message=f"Upload {filename}")

    @classmethod
    def from_model(cls, model, reference, *, device: str = "auto") -> "Aligner":
        """Wrap an already-loaded ``(model, reference)`` pair (e.g. straight from training)."""
        dev = resolve_device(device)
        return cls(model.to(dev), reference, device=dev)

    def bound_to(self, reference) -> "Aligner":
        """An Aligner that predicts from ``reference`` (its bank, e.g. a donor genotype's subset of the reference): a
        copy of the model bound to it once, so later calls, and the workers a pool starts from it, use that bank
        with no per-call binding; the reference is annotated against this Aligner's own (``alignair.reference.
        annotate``), as one given at load is. This Aligner is unchanged. The copy is not a model file (``model_path`` is unset),
        and its label schema names the bank's alleles."""
        import copy

        from .reference import annotate
        reference = annotate(reference, known=self.reference)   # as a reference given at load is
        model = copy.deepcopy(self.model)
        model.bind_reference(reference)
        return Aligner(model, reference, device=self.device, source_commit=self.source_commit)

    # ------------------------------------------------------------------ properties
    @property
    def label_schema(self) -> dict:
        """The identity of ``predict_raw``'s allele columns: each gene's alleles in the bound reference's order."""
        cfg = self.model.cfg
        genes = {gene: list(self.reference.gene(gene.upper()).names) for gene in cfg.genes}
        return {"schema_version": 1, "input_space": cfg.input_space, "label_space": cfg.label_space, "genes": genes}

    @property
    def loci(self) -> tuple:
        """The bound reference's AIRR locus codes, in order (empty if it declares none)."""
        return self.reference.locus_names()

    @property
    def default_locus(self) -> str:
        """The locus written for a record that carries none: the reference's one locus, else IGH."""
        return self.reference.default_locus

    # ------------------------------------------------------------------ inference
    def predict(self, sequences: Sequence[str], *, batch_size="auto", genotype=None, airr: bool = True,
                **overrides) -> PredictionResult:
        """Eagerly align a list of nucleotide strings, returning a :class:`PredictionResult`.
        ``batch_size="auto"`` picks a device-appropriate size. ``genotype`` (a ``{V|D|J: allele names}`` dict) is the
        bank the calls come from: a subset of the bound reference."""
        from .api import predict_sequences
        bs = _auto_batch_size(self.device) if batch_size in (None, "auto") else int(batch_size)
        if genotype is not None:
            overrides["genotype"] = genotype
        records = predict_sequences(self.model, self.reference, sequences, device=self.device,
                                    batch_size=bs, airr=airr, **overrides)
        return PredictionResult(records, locus=self.default_locus)

    def predict_raw(self, sequences, *, batch_size=64) -> dict:
        """The model's raw head arrays; label_schema describes column identity.

        This method does not run the AIRR assembly. Every output uses model semantics (allele scores relative to the
        read's best allele, model-frame positions). The forward is the one ``predict`` runs (``predict.forward.
        run_model``: reads batched by width bucket), so the arrays are the ones a read's row comes from.
        """
        import numpy as np
        from .core.setup import _input_config
        from .predict import PredictConfig
        from .predict.forward import run_model
        cfg = self.model.cfg
        _input_config(cfg)
        batches = run_model(self.model, sequences, PredictConfig(max_seq_length=cfg.max_seq_length,
                                                                 batch_size=batch_size), self.device)
        return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}

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

