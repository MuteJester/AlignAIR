"""Catalog of GenAIRR capabilities used by benchmark generation."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

ALLOWED_GENAIRR_FEATURE_STATUSES = ("integrated", "partial", "planned")


@dataclass(frozen=True)
class GenAIRRFeature:
    """One GenAIRR capability and how the benchmark currently uses it."""

    name: str
    status: str
    genairr_entry_points: tuple[str, ...]
    benchmark_surface: tuple[str, ...]
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


GENAIRR_FEATURES: tuple[GenAIRRFeature, ...] = (
    GenAIRRFeature(
        name="vdj_recombination",
        status="integrated",
        genairr_entry_points=("Experiment.on", "Experiment.recombine"),
        benchmark_surface=("StratumSpec.progress", "stratum param_overrides"),
        notes="Core record generation path for all benchmark cases.",
    ),
    GenAIRRFeature(
        name="productive_only",
        status="integrated",
        genairr_entry_points=("Experiment.productive_only",),
        benchmark_surface=("productive_only_clean", "productive_only param override"),
    ),
    GenAIRRFeature(
        name="s5f_mutation",
        status="integrated",
        genairr_entry_points=("Experiment.mutate",),
        benchmark_surface=("mutation_rate", "mutation_count", "high_shm strata"),
        notes="Benchmark can use rate or explicit count distributions through stratum params.",
    ),
    GenAIRRFeature(
        name="end_loss_trimming",
        status="integrated",
        genairr_entry_points=("Experiment.end_loss_5prime", "Experiment.end_loss_3prime"),
        benchmark_surface=("trimmed", "extreme_end_loss", "end_loss_5/end_loss_3 overrides"),
    ),
    GenAIRRFeature(
        name="polymerase_indels",
        status="integrated",
        genairr_entry_points=("Experiment.polymerase_indels",),
        benchmark_surface=("high_indel", "high_indel_extreme", "indel_count overrides"),
    ),
    GenAIRRFeature(
        name="sequencing_errors",
        status="integrated",
        genairr_entry_points=("Experiment.sequencing_errors",),
        benchmark_surface=("seq_error_rate", "noisy_ambiguous", "ambiguous_n_extreme"),
    ),
    GenAIRRFeature(
        name="ambiguous_base_calls",
        status="integrated",
        genairr_entry_points=("Experiment.ambiguous_base_calls",),
        benchmark_surface=("ambiguous_count", "noisy_ambiguous", "ambiguous_n_extreme"),
    ),
    GenAIRRFeature(
        name="d_inversion",
        status="integrated",
        genairr_entry_points=("Experiment.invert_d",),
        benchmark_surface=("forced_d_inversion", "invert_d_prob override"),
    ),
    GenAIRRFeature(
        name="receptor_revision",
        status="integrated",
        genairr_entry_points=("Experiment.receptor_revision",),
        benchmark_surface=("receptor_revision", "receptor_revision_prob override"),
    ),
    GenAIRRFeature(
        name="contamination",
        status="integrated",
        genairr_entry_points=("Experiment.contaminate",),
        benchmark_surface=("contaminant", "contaminate_prob override"),
    ),
    GenAIRRFeature(
        name="paired_end_reads",
        status="integrated",
        genairr_entry_points=("Experiment.paired_end",),
        benchmark_surface=("paired_end", "read_layout coverage labels"),
    ),
    GenAIRRFeature(
        name="adaptive_anchor_amplicons",
        status="integrated",
        genairr_entry_points=("GenAIRR AIRR coordinates",),
        benchmark_surface=("adaptive_igh_strata", "apply_benchmark_crop"),
        notes="Benchmark-side crop policy models FR/J anchored amplicons while preserving GenAIRR truth.",
    ),
    GenAIRRFeature(
        name="deterministic_orientation_transforms",
        status="integrated",
        genairr_entry_points=("GenAIRR forward AIRR records",),
        benchmark_surface=("orientation_ids", "presented_genes", "presented labels"),
        notes="Orientation is applied after GenAIRR so canonical and presented truth are both retained.",
    ),
    GenAIRRFeature(
        name="strict_sampling",
        status="partial",
        genairr_entry_points=("stream_records(strict=...)", "run_records(strict=...)"),
        benchmark_surface=("strict_sampling param override",),
        notes="Generation can request strict stream sampling; CI-level record validation remains planned.",
    ),
    GenAIRRFeature(
        name="segment_targeted_mutation",
        status="partial",
        genairr_entry_points=("Experiment.mutate(segment_rates=...)", "Experiment.mutate(v_subregion_rates=...)"),
        benchmark_surface=("segment_rates/v_subregion_rates param overrides",),
        notes="Supported by the shared experiment builder; dedicated assay strata are still planned.",
    ),
    GenAIRRFeature(
        name="pcr_substitution_errors",
        status="partial",
        genairr_entry_points=("Experiment.pcr_amplify",),
        benchmark_surface=("pcr_error_rate/pcr_error_count param overrides",),
        notes="Supported by the shared experiment builder; dedicated PCR-error strata are still planned.",
    ),
    GenAIRRFeature(
        name="metadata_stamping",
        status="partial",
        genairr_entry_points=("Experiment.with_metadata",),
        benchmark_surface=("metadata param override",),
        notes="Supported as a param pass-through; subject/cohort metadata recipes are still planned.",
    ),
    GenAIRRFeature(
        name="allele_restriction",
        status="partial",
        genairr_entry_points=("Experiment.restrict_alleles",),
        benchmark_surface=("restrict_alleles param override", "rare allele and missing allele strata"),
        notes="Builder pass-through exists; curated rare/missing-allele assay strata are still planned.",
    ),
    GenAIRRFeature(
        name="genotype_and_cohort_sampling",
        status="partial",
        genairr_entry_points=("Experiment.with_genotype", "Experiment.run_cohort"),
        benchmark_surface=("genotype_seed/genotype_subject_id param overrides", "genotype_subject_spec"),
        notes="Single-subject genotype strata are available; true run_cohort orchestration remains planned.",
    ),
    GenAIRRFeature(
        name="clonal_repertoire_and_lineage",
        status="planned",
        genairr_entry_points=("Experiment.clonal_repertoire", "Experiment.clonal_lineage"),
        benchmark_surface=("clone-aware benchmark recipes",),
    ),
    GenAIRRFeature(
        name="record_validation_gate",
        status="partial",
        genairr_entry_points=("run_records(validate_records=True)", "SimulationResult.validate_records"),
        benchmark_surface=("validate_records param override", "validate_stratum_records"),
        notes="Available for bounded generation/preflight; large streaming builds still default to zero-overhead streaming.",
    ),
    GenAIRRFeature(
        name="multi_locus_configs",
        status="partial",
        genairr_entry_points=("GenAIRR.data",),
        benchmark_surface=("multi_locus_specs",),
        notes="Per-DataConfig locus probes are available; unified cross-locus reporting remains planned.",
    ),
)


def genairr_feature_catalog() -> list[dict[str, Any]]:
    """Return the benchmark's GenAIRR capability integration catalog."""

    return [feature.to_dict() for feature in GENAIRR_FEATURES]


def validate_genairr_feature_catalog() -> dict[str, Any]:
    """Validate catalog uniqueness and status vocabulary."""

    errors: list[str] = []
    names = [feature.name for feature in GENAIRR_FEATURES]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate GenAIRR feature names: {duplicates}")
    for feature in GENAIRR_FEATURES:
        if feature.status not in ALLOWED_GENAIRR_FEATURE_STATUSES:
            errors.append(f"{feature.name}: invalid status {feature.status!r}")
        if not feature.genairr_entry_points:
            errors.append(f"{feature.name}: missing GenAIRR entry points")
        if not feature.benchmark_surface:
            errors.append(f"{feature.name}: missing benchmark surface")
    return {
        "valid": not errors,
        "errors": errors,
        "n_features": len(GENAIRR_FEATURES),
        "statuses": dict(
            sorted(
                (status, sum(1 for feature in GENAIRR_FEATURES if feature.status == status))
                for status in ALLOWED_GENAIRR_FEATURE_STATUSES
            )
        ),
    }
