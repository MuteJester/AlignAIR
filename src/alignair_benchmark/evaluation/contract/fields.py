from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.schema import GENES

LEVELS = ("minimal", "core", "assay")


@dataclass(frozen=True)
class PredictionField:
    """One prediction field accepted by benchmark scorers."""

    name: str
    dtype: str
    level: str
    description: str
    required_if_has_d: bool = False
    aliases: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _gene_fields(name: str, dtype: str, level: str, description: str) -> tuple[PredictionField, ...]:
    return tuple(
        PredictionField(
            name=f"{gene}_{name}",
            dtype=dtype,
            level=level,
            description=description.format(gene=gene.upper()),
            required_if_has_d=(gene == "d"),
        )
        for gene in GENES
    )


PREDICTION_FIELDS: tuple[PredictionField, ...] = (
    PredictionField("sequence_id", "str", "assay", "Stable input sequence identifier."),
    PredictionField("sequence", "str", "assay", "Presented sequence used by the aligner."),
    PredictionField("locus", "str", "assay", "Locus or chain label, for example IGH."),
    PredictionField("orientation_id", "int[0..3]", "core", "Presented-read orientation class."),
    *_gene_fields("call", "str", "minimal", "Top {gene} allele call."),
    *_gene_fields("calls", "list[str]", "core", "Set-valued {gene} allele call."),
    *_gene_fields("sequence_start", "int", "core", "{gene} start in the evaluated query frame."),
    *_gene_fields("sequence_end", "int", "core", "{gene} end in the evaluated query frame."),
    *_gene_fields("germline_start", "int", "core", "{gene} start in the called germline allele."),
    *_gene_fields("germline_end", "int", "core", "{gene} end in the called germline allele."),
    *_gene_fields("cigar", "str", "assay", "{gene} alignment CIGAR."),
    *_gene_fields("identity", "float", "assay", "{gene} identity/supporting alignment quality."),
    *_gene_fields("trim_5", "int", "assay", "{gene} 5-prime trimming."),
    *_gene_fields("trim_3", "int", "assay", "{gene} 3-prime trimming."),
    *_gene_fields("ranked_calls", "list[str]", "assay", "Ranked {gene} candidate list."),
    *_gene_fields("resolved_call", "str", "assay",
                  "Hierarchically resolved {gene} call (allele/gene/family) or null if abstained."),
    *_gene_fields("call_level", "str", "assay",
                  "Resolution level of {gene}_resolved_call: allele|gene|family|none."),
    PredictionField("productive", "bool|float", "core", "Predicted productivity status."),
    PredictionField("vj_in_frame", "bool", "assay", "Whether V/J are in frame."),
    PredictionField("stop_codon", "bool", "assay", "Whether the translated sequence contains a stop codon."),
    PredictionField("junction", "str", "assay", "Junction/CDR3 nucleotide sequence."),
    PredictionField("junction_aa", "str", "assay", "Junction/CDR3 amino-acid sequence."),
    PredictionField("junction_start", "int", "assay", "Junction/CDR3 start in the evaluated query frame."),
    PredictionField("junction_end", "int", "assay", "Junction/CDR3 end in the evaluated query frame."),
    PredictionField("junction_length", "int", "assay", "Junction/CDR3 nucleotide length."),
    PredictionField("np1", "str", "assay", "N/P sequence between V and D."),
    PredictionField("np2", "str", "assay", "N/P sequence between D and J."),
    PredictionField("np1_length", "int", "assay", "Length of the V-D N/P region."),
    PredictionField("np2_length", "int", "assay", "Length of the D-J N/P region."),
    PredictionField("p_region_length", "int", "assay", "Total P-nucleotide length when reported directly."),
    PredictionField("d_inverted", "bool", "assay", "Whether D was called in inverted orientation."),
    PredictionField("c_call", "str", "assay", "Constant-region allele/gene call when applicable."),
    PredictionField("region_labels", "list[int]", "assay", "Per-position region labels in benchmark encoding."),
    PredictionField("state_labels", "list[int]", "assay", "Per-position edit-state labels in benchmark encoding."),
    PredictionField("mutation_rate", "float", "assay", "Predicted SHM mutation rate."),
    PredictionField("noise_count", "float", "assay", "Predicted sequencing/PCR noise count."),
    PredictionField("indel_count", "float", "assay", "Predicted indel burden."),
    PredictionField("read_layout", "str", "assay", "Read-layout label, for example paired_end."),
    PredictionField("is_contaminant", "bool", "assay", "Whether the input is a contaminant/out-of-scope read."),
    PredictionField("receptor_revision_applied", "bool", "assay", "Whether receptor revision was detected."),
)


def prediction_contract() -> list[dict[str, Any]]:
    """Return the benchmark prediction field contract as serializable data."""

    return [field.to_dict() for field in PREDICTION_FIELDS]


def _expected_fields(level: str, has_d: bool) -> tuple[PredictionField, ...]:
    if level not in LEVELS:
        raise ValueError(f"level must be one of {LEVELS}")
    max_idx = LEVELS.index(level)
    fields = []
    for field in PREDICTION_FIELDS:
        if LEVELS.index(field.level) > max_idx:
            continue
        if field.name.startswith("d_") and not has_d:
            continue
        fields.append(field)
    return tuple(fields)
