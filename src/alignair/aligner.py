"""The inference surface: load a model (``Aligner.from_pretrained``) and predict, a typed and registry-aware wrapper
over the prediction pipeline (``alignair.predict``).

    from alignair import Aligner
    # a catalog id (PENDING-V4-PUBLISH), a path, or id@version
    aligner = Aligner.from_pretrained("alignair4-igh-human", device="auto")
    result = aligner.predict(["CAGGTGCAGCTG..."])
    result.write_airr("predictions.tsv")

Training is ``alignair.run_training`` (``alignair.train``).
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .core.runtime import resolve_device
from .errors import ConfigurationError, InputError, MissingDependencyError, ModelError
from .predict import PredictConfig
from .predict import predict as _predict
from .reference.loci import FALLBACK_LOCUS

if TYPE_CHECKING:
    from .core.model import AlignAIR
    from .reference.reference_set import ReferenceSet

__all__ = ["Aligner", "PredictionResult", "resolve_device"]


def auto_batch_size(device: str, workers: int = 1) -> int:
    """256 on a GPU, 64 in one CPU process, 16 per single-threaded CPU worker (``alignair.parallel``), whose
    activations then stay in its share of the cache."""
    if device.startswith("cuda"):
        return 256
    return 16 if workers > 1 else 64


@dataclass
class PredictionResult:
    """A typed prediction result: the AIRR-convention records plus convenience I/O. Iterable and
    ``len()``-able over the per-read records; ``write_airr`` emits the TSV; ``to_dicts`` returns the raw
    list. The coordinates in each record refer to that record's canonical ``sequence``."""
    records: list[dict]
    locus: str = FALLBACK_LOCUS
    ids: list[str] | None = None

    def __iter__(self) -> Iterator[dict]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    def to_dicts(self) -> list:
        """The records as a list of dicts, one per read."""
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
    """A loaded model bound to its germline reference, ready to predict. Construct with :meth:`from_pretrained` (a
    local path, a shipped catalog id, or ``id@revision``); ``alignair predict`` loads its model through it."""

    def __init__(self, model: AlignAIR, reference: ReferenceSet, *, device: str = "auto",
                 model_path: str | None = None, source_commit: str | None = None) -> None:
        self.device = resolve_device(device)    # "auto": CUDA, then Apple MPS, then CPU
        self.model = model.to(self.device)
        self.reference = reference
        self.model_path = model_path
        self.source_commit = source_commit     # resolved HF commit SHA (provenance), else None

    # ------------------------------------------------------------------ constructors
    @classmethod
    def from_pretrained(cls, model: str, *, revision: str | None = None, device: str = "auto",
                        reference: ReferenceSet | None = None, registry: str | Iterable[str] | None = None,
                        offline: bool = False,
                        token: str | None = None) -> Aligner:
        """Load a model by local path, a Hugging Face repo (``hf://org/repo`` or ``org/repo`` — pulled
        via ``huggingface_hub``), or a shipped catalog id (``id`` / ``id@version``), looked up in the registries of
        ``registry`` (a string is one registry; default: the configured ones). ``revision`` pins a
        HF branch/tag/commit (or catalog version); ``token`` authenticates a private/gated HF repo (or
        ``HF_TOKEN``). ``offline`` uses only local caches. ``device`` accepts ``"auto"`` (CUDA→MPS→CPU)."""
        from .model_file import load_model
        from .registry import hf, resolve_model
        path = str(resolve_model(model, registry=registry, offline=offline, token=token, revision=revision))
        dev = resolve_device(device)
        loaded = load_model(path, reference=reference, device=dev)
        return cls(loaded.model, loaded.reference, device=dev, model_path=path, source_commit=hf.resolved_commit(path))

    def _assert_distributable(self) -> None:
        """Refuse to distribute a resumable checkpoint: those carry trusted **pickle** sections
        (dataconfig / train_state) and must not be published — export a pickle-free inference artifact
        first."""
        if not self.model_path:
            raise ModelError("needs an aligner loaded from a file (model_path is unset)")
        from .model_file import pickle_sections, read_metadata
        pickle_secs = pickle_sections(read_metadata(self.model_path))
        if pickle_secs:
            raise ModelError(
                f"{self.model_path} is a resumable checkpoint with pickle sections {pickle_secs}; it is "
                f"not distributable. Export a pickle-free bundle first: `alignair export {self.model_path} "
                f"<out_dir>` (alignair.train.bundle.export_bundle), then share <out_dir>/model.alignair.")

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
        the shared catalog repo, use ``alignair.registry.publish``'s ``publish_bundle`` and ``upload_version``.
        Returns the commit."""
        import os
        self._assert_distributable()  # never upload a pickle checkpoint
        try:
            import huggingface_hub
        except ImportError as e:                    # pragma: no cover - optional dep
            raise MissingDependencyError("push_to_hub needs 'huggingface_hub' (`pip install alignair[hub]`)") from e
        from .registry.hf import hf_token
        api = huggingface_hub.HfApi(token=hf_token(token))
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
    def from_model(cls, model: AlignAIR, reference: ReferenceSet, *, device: str = "auto") -> Aligner:
        """Wrap an already-loaded ``(model, reference)`` pair (e.g. straight from training), the model placed on
        ``device``."""
        return cls(model, reference, device=device)

    def bound_to(self, reference: ReferenceSet) -> Aligner:
        """An Aligner that predicts from ``reference`` (its bank, e.g. a donor genotype's subset of the reference): a
        copy of the model bound to it once, so later calls, and the workers a pool starts from it, use that bank
        with no per-call binding; the reference is annotated against this Aligner's own (``alignair.reference.
        annotate``), as one given at load is. This Aligner is unchanged. The copy is not a model file (``model_path``
        is unset), and its label schema names the bank's alleles."""
        import copy

        from .reference import annotate
        reference = annotate(reference, known=self.reference)   # as a reference given at load is
        model = copy.deepcopy(self.model)
        model.bind_reference(reference)
        return Aligner(model, reference, device=self.device, source_commit=self.source_commit)

    def with_genotype(self, genotype: Mapping[str, Iterable[str]]) -> Aligner:
        """An Aligner whose calls come from a donor's ``genotype`` (``{V|D|J: allele names}``, either case): this
        Aligner bound once to the genotype's subset of its reference (``bound_to``); a gene it does not name keeps all
        its alleles. This Aligner is unchanged. UnknownAlleleError for an allele the reference does not hold (give its
        sequence in the reference), InputError for a gene left with no allele, ConfigurationError for a key that is
        not V, D or J."""
        from .genotype.bank import genotype_bank
        return self.bound_to(genotype_bank(self.reference, genotype))

    # ------------------------------------------------------------------ properties
    @property
    def label_schema(self) -> dict:
        """The identity of ``predict_raw``'s allele columns: each gene's alleles in the bound reference's order."""
        genes = {gene: list(self.reference.gene(gene.upper()).names) for gene in self.model.cfg.genes}
        return {"schema_version": 1, "genes": genes}

    @property
    def loci(self) -> tuple:
        """The bound reference's AIRR locus codes, in order (empty if it declares none)."""
        return self.reference.locus_names()

    @property
    def default_locus(self) -> str:
        """The locus written for a record that carries none: the reference's one locus, else IGH."""
        return self.reference.default_locus

    # ------------------------------------------------------------------ inference
    def _batch_size(self, batch_size: int | str) -> int | str:
        return auto_batch_size(self.device) if batch_size == "auto" else batch_size

    def predict(self, sequences: Sequence[str], *, ids: Sequence[str] | None = None, batch_size: int | str = "auto",
                threshold: float | None = None, cap: int | None = None, airr: bool = True) -> PredictionResult:
        """Align nucleotide strings, returning a :class:`PredictionResult` of one record per sequence, in order: every
        sequence given is predicted (``read_sequences`` and the CLI drop the reads of a file that are mostly not
        bases; an empty one is refused here).

        ``ids`` (one string per sequence) fill each record's ``sequence_id`` and the result's ``ids``.
        ``batch_size="auto"`` picks a device-appropriate size. ``threshold`` and ``cap`` set the called alleles of each
        gene (``v_calls``...): those scoring at least ``threshold`` relative to the best (default 0.5), at most ``cap``
        of them (default 3), never none. ``airr=False`` returns light records without the AIRR assembly. To call from a
        donor's genotype, predict with :meth:`with_genotype`'s Aligner."""
        if ids is not None:
            sequences, ids = _named_reads(sequences, ids)
        cfg = PredictConfig.for_model(self.model, self.reference, batch_size=self._batch_size(batch_size),
                                      threshold=threshold, cap=cap, airr=airr)
        records = _predict(self.model, sequences, self.reference, cfg, device=self.device)
        if ids is not None:
            records = [{"sequence_id": name, **record} for name, record in zip(ids, records)]
        return PredictionResult(records, locus=self.default_locus, ids=ids)

    def predict_raw(self, sequences: Sequence[str], *, batch_size: int | str = "auto") -> dict:
        """The model's raw head arrays; label_schema describes column identity.

        This method does not run the AIRR assembly. Every output uses model semantics (allele scores relative to the
        read's best allele, model-frame positions). The forward is the one ``predict`` runs (``predict.forward.
        run_model``: reads batched by width bucket), so the arrays are the ones a read's row comes from.
        """
        import numpy as np

        from .predict.forward import run_model
        cfg = self.model.cfg
        batches = run_model(self.model, sequences, PredictConfig(max_seq_length=cfg.max_seq_length,
                                                                 batch_size=self._batch_size(batch_size)), self.device)
        return {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}

    def predict_iter(self, sequences: Iterable[str], *, ids: Iterable[str] | None = None,
                     batch_size: int | str = "auto", chunk_size: int = 20000, threshold: float | None = None,
                     cap: int | None = None, airr: bool = True) -> Iterator[dict]:
        """Stream prediction over an iterable of sequences in bounded memory: yields per-read records, predicting
        ``chunk_size`` reads at a time (a positive integer). Same options as :meth:`predict`; ``ids`` may be an
        iterable too (InputError when it runs out before the sequences, or after)."""
        from itertools import islice, zip_longest
        from numbers import Integral

        from .data.inputs import sequence_batch_iterator
        if isinstance(chunk_size, bool) or not isinstance(chunk_size, Integral) or chunk_size < 1:
            raise ConfigurationError(f"chunk_size must be a positive integer (got {chunk_size!r})",
                                     context={"field": "chunk_size"})
        reads = sequence_batch_iterator(sequences)
        names = _id_iterator(ids) if ids is not None else None
        options = {"batch_size": batch_size, "threshold": threshold, "cap": cap, "airr": airr}
        while True:
            if names is None:
                chunk, chunk_ids = list(islice(reads, chunk_size)), None
            else:
                pairs = list(islice(zip_longest(reads, names, fillvalue=_MISSING), chunk_size))
                if any(_MISSING in pair for pair in pairs):
                    raise InputError("ids and sequences differ in number", context={"field": "ids"})
                chunk, chunk_ids = [read for read, _ in pairs], [name for _, name in pairs]
            if not chunk:
                return
            yield from self.predict(chunk, ids=chunk_ids, **options).records


_MISSING = object()


def _id_iterator(ids) -> Iterator:
    """An iterator over ``ids``; InputError for a bare string, bytes or mapping, or a non-iterable."""
    if isinstance(ids, (str, bytes, Mapping)):
        raise InputError("ids must be a sequence of strings, one per sequence, not a bare string, bytes or mapping",
                         context={"field": "ids"})
    try:
        return iter(ids)
    except TypeError as exc:
        raise InputError("ids must be a sequence of strings, one per sequence", context={"field": "ids"}) from exc


def _named_reads(sequences, ids) -> tuple[list, list]:
    """``(sequences, ids)`` as lists; InputError unless ``ids`` is one string per sequence."""
    from .data.inputs import sequence_batch_iterator
    reads, names = list(sequence_batch_iterator(sequences)), list(_id_iterator(ids))
    if len(names) != len(reads):
        raise InputError(f"{len(names)} ids for {len(reads)} sequences: give one id per sequence",
                         context={"field": "ids"})
    if not all(isinstance(name, str) for name in names):
        raise InputError("ids must be strings", context={"field": "ids"})
    return reads, names

