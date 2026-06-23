"""Stable public Python API for AlignAIR.

The CLI is a client of this module — use the same functions from notebooks / pipelines:

    from alignair import load_model, predict, read_sequences, write_airr, ReferenceSet

    model = load_model("runs/my_model/bundle", device="cuda")
    ids, reads, _ = read_sequences("reads.fastq")
    result = predict(model, reads, genotype="donor.yaml")     # genotype optional
    result.to_airr("out.tsv", ids)

Public surface lives under ``alignair.*``; everything under ``alignair.inference``,
``alignair.io``, ``alignair.serialization``, etc. is implementation detail and may change.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class LoadedModel:
    """A loaded model plus its default reference, calibration, and declared locus."""
    model: object
    reference_set: Optional[object]      # default/embedded reference (None for a bare checkpoint)
    dataconfigs: Optional[List[str]]
    locus: str
    calibration: Optional[dict]
    device: str
    has_d: Optional[bool] = None


@dataclass
class PredictionBatch:
    """Result of :func:`predict`. ``sequences`` are the CANONICAL (forward) sequences the
    coordinates in ``predictions`` refer to (reverse-complemented inputs are reoriented)."""
    sequences: List[str]
    predictions: List[dict]              # one AIRR-ready dict per read (v/d/j calls, coords, ...)
    locus: str
    extra_columns: List[str] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.predictions)

    def to_airr(self, path: str, ids: List[str]) -> None:
        """Write the AIRR rearrangement TSV (``path`` may be '-' for stdout)."""
        from .io.airr import write_airr
        write_airr(path, ids, self.sequences, self.predictions, locus=self.locus)


def load_model(spec: str, device: Optional[str] = None) -> LoadedModel:
    """Load a model from a bundle dir, a raw .pt checkpoint, a catalog id, or an ``org/name``
    Hugging Face repo id (auto-downloaded). Returns a :class:`LoadedModel`."""
    import torch
    from .hub import resolve_model
    from .serialization.dnalignair_bundle import is_bundle, load_dnalignair_bundle
    from .reference.reference_set import ReferenceSet
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    path = resolve_model(spec)
    if is_bundle(path):
        b = load_dnalignair_bundle(path, build=True, device=device)
        rs = b.get("reference_set")
        if rs is None and b.get("dataconfigs"):
            import GenAIRR.data as gdata
            rs = ReferenceSet.from_dataconfigs(*[getattr(gdata, n) for n in b["dataconfigs"]])
        return LoadedModel(b["model"], rs, b.get("dataconfigs"), b.get("locus", "IGH"),
                           b.get("calibration"), device, has_d=(rs.has_d if rs else None))
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    ckpt = torch.load(path, map_location=device)
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return LoadedModel(model, None, None, "IGH", None, device)


def _resolve_reference(genotype, reference, loaded: LoadedModel):
    from .reference.reference_set import ReferenceSet
    if reference is not None:
        return reference
    if genotype is not None:
        if isinstance(genotype, ReferenceSet):
            return genotype
        if isinstance(genotype, dict):
            return ReferenceSet.from_genotype(genotype)
        import os
        ext = os.path.splitext(str(genotype))[1].lower()
        return (ReferenceSet.from_fasta(genotype) if ext in (".fasta", ".fa", ".fna", ".faa")
                else ReferenceSet.from_yaml(genotype))
    if loaded.reference_set is not None:
        return loaded.reference_set
    raise ValueError("no reference available: pass reference=/genotype=, or load a bundle that "
                     "carries a default reference")


def predict(model: LoadedModel, reads: List[str], *, reference=None, genotype=None,
            batch_size: int = 64, v_reader: str = "learned", full_alignment: bool = True,
            calibration=None, locus: Optional[str] = None, progress: bool = False) -> PredictionBatch:
    """Align ``reads`` (a list of nucleotide strings) with a :class:`LoadedModel`.

    Reference precedence: explicit ``reference`` (a ReferenceSet) > ``genotype`` (a ReferenceSet,
    dict, or YAML/FASTA path) > the model's default/embedded reference. ``calibration`` defaults to
    the bundle's. ``full_alignment`` adds exact cigars / germline_alignment / identity (needs
    parasail; falls back to coordinate cigars otherwise)."""
    from .inference.dnalignair_infer import predict_reads, canonicalize_sequence
    rs = _resolve_reference(genotype, reference, model)
    cal = calibration if calibration is not None else model.calibration
    preds = predict_reads(model.model, rs, reads, device=model.device, batch_size=batch_size,
                          rerank="learned", v_reader=v_reader, calibration=cal,
                          full_alignment=full_alignment, progress=progress)
    canon = [canonicalize_sequence(s, p["orientation_id"]) for s, p in zip(reads, preds)]
    out_locus = locus or rs.infer_locus() or model.locus or "IGH"
    return PredictionBatch(canon, preds, out_locus)
