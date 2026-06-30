"""Scoring component metadata catalog.

The scorer manifest is the code-owned contract for what each scorer can emit.
The criteria catalog explains why metrics matter; this manifest explains where
they come from and how they are aggregated.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ....nn.heads.region import REGIONS
from ....nn.heads.state import STATES
from ..performance import PERFORMANCE_GLOBAL_KEYS, PERFORMANCE_PREDICTION_FIELD_KEYS

AGGREGATION_MEAN = "mean"


@dataclass(frozen=True)
class ScoringComponentManifest:
    """Metadata for one scoring component."""

    name: str
    scope: str
    metric_keys: tuple[str, ...]
    aggregation: str = AGGREGATION_MEAN
    required_prediction_fields: tuple[str, ...] = ()
    ground_truth_fields: tuple[str, ...] = ()
    scenario_axes: tuple[str, ...] = ()
    description: str = ""

    def to_dict(self, *, include_metric_specs: bool = True) -> dict[str, Any]:
        data = asdict(self)
        if include_metric_specs:
            data["metrics"] = [_metric_row(key, self.aggregation) for key in self.metric_keys]
        return data


def _metric_row(key: str, aggregation: str) -> dict[str, Any]:
    from ...core import metric_registry, metric_spec

    spec = metric_spec(key)
    return {
        "key": key,
        "aggregation": aggregation,
        "registered": key in metric_registry(),
        "higher_is_better": spec.higher_is_better,
        "direction": "higher_is_better" if spec.higher_is_better else "lower_is_better",
        "pass_threshold": spec.pass_threshold,
        "warn_threshold": spec.warn_threshold,
        "criteria": spec.criterion_names,
        "categories": spec.categories,
        "statuses": spec.statuses,
        "importance": spec.importance,
    }


def _region_metric_keys() -> tuple[str, ...]:
    per_class = tuple(
        f"region_{label.lower()}_{suffix}"
        for label in REGIONS
        for suffix in ("recall", "f1")
    )
    return ("region_acc", "region_per_class_recall", "region_macro_f1", *per_class)


def _state_metric_keys() -> tuple[str, ...]:
    per_class = tuple(
        f"state_{label.lower()}_{suffix}"
        for label in STATES
        for suffix in ("recall", "f1")
    )
    event_metrics = (
        "substitution_precision",
        "substitution_recall",
        "substitution_f1",
        "insertion_precision",
        "insertion_recall",
        "insertion_f1",
        "deletion_precision",
        "deletion_recall",
        "deletion_f1",
        "shm_site_precision",
        "shm_site_recall",
        "shm_site_f1",
        "false_shm_from_noise_rate",
        "indel_event_precision",
        "indel_event_recall",
        "indel_event_f1",
    )
    return ("state_acc", "state_per_class_recall", "state_macro_f1", *per_class, *event_metrics)


def _coordinate_metric_keys() -> tuple[str, ...]:
    return tuple(
        f"{suffix}_{metric}"
        for suffix in ("ss", "se", "gs", "ge")
        for metric in ("mae", "exact", "within1", "within3", "within10")
    )


ORIENTATION_MANIFEST = ScoringComponentManifest(
    name="orientation",
    scope="global",
    metric_keys=("orientation_acc",),
    required_prediction_fields=("orientation_id",),
    ground_truth_fields=("orientation_id", "rev_comp"),
    scenario_axes=("orientation",),
    description="Presented-read orientation detection.",
)

AIRR_CONTRACT_MANIFEST = ScoringComponentManifest(
    name="airr_contract",
    scope="global",
    metric_keys=("required_field_presence", "optional_field_presence", "parseable_airr_rate"),
    required_prediction_fields=("sequence_id", "sequence", "v_call", "j_call", "productive", "junction"),
    ground_truth_fields=("sequence_id", "sequence", "v_call", "d_call", "j_call", "productive", "junction"),
    scenario_axes=("difficulty_stratum", "locus_chain"),
    description="AIRR-style output completeness and parseability.",
)

SEGMENT_ORDER_MANIFEST = ScoringComponentManifest(
    name="segment_order",
    scope="global",
    metric_keys=("vdj_order_valid", "overlap_rate", "negative_span_rate"),
    required_prediction_fields=("v_sequence_start/end", "d_sequence_start/end", "j_sequence_start/end"),
    ground_truth_fields=("v_sequence_start/end", "d_sequence_start/end", "j_sequence_start/end"),
    scenario_axes=("length", "segment_presence", "orientation"),
    description="V/D/J ordering, overlap, and negative-span consistency.",
)

JUNCTION_MANIFEST = ScoringComponentManifest(
    name="junction",
    scope="global",
    metric_keys=(
        "junction_start_mae",
        "junction_end_mae",
        "cdr3_overlap_iou",
        "junction_nt_exact",
        "junction_aa_exact",
        "junction_length_mae",
        "junction_aa_length_mae",
        "np1_exact",
        "n1_length_mae",
        "np1_length_mae",
        "np2_exact",
        "n2_length_mae",
        "np2_length_mae",
        "p_region_length_mae",
    ),
    required_prediction_fields=("junction/cdr3", "junction_aa/cdr3_aa", "junction_start/end", "np1/np2"),
    ground_truth_fields=("junction", "junction_aa", "junction_start", "junction_end", "np1", "np2"),
    scenario_axes=("junction_biology", "length", "segment_presence"),
    description="Junction/CDR3 and N/P-addition region scoring.",
)

METADATA_MANIFEST = ScoringComponentManifest(
    name="metadata",
    scope="global",
    metric_keys=(
        "locus_acc",
        "chain_type_acc",
        "has_d_routing_acc",
        "c_found_rate",
        "c_call_acc",
        "c_gene_acc",
        "d_inversion_acc",
        "vj_in_frame_acc",
        "stop_codon_acc",
        "fwr1_mutation_count_mae",
        "cdr1_mutation_count_mae",
        "fwr2_mutation_count_mae",
        "cdr2_mutation_count_mae",
        "fwr3_mutation_count_mae",
        "layout_specific_call_acc",
        "contaminant_no_call_rate",
        "contaminant_handled_rate",
        "false_positive_alignment_rate",
        "contaminant_flag_acc",
        "revision_flag_acc",
        "revision_case_call_acc",
        "revision_case_boundary_mae",
    ),
    required_prediction_fields=("locus/chain_type", "c_call", "d_inverted", "vj_in_frame", "stop_codon"),
    ground_truth_fields=("locus", "c_call", "d_inverted", "vj_in_frame", "stop_codon", "is_contaminant"),
    scenario_axes=("locus_chain", "d_orientation", "read_layout", "productivity", "input_validity"),
    description="Locus, biological flags, contaminant handling, and revision-case scoring.",
)

PERFORMANCE_MANIFEST = ScoringComponentManifest(
    name="performance",
    scope="global",
    metric_keys=PERFORMANCE_GLOBAL_KEYS,
    required_prediction_fields=PERFORMANCE_PREDICTION_FIELD_KEYS,
    scenario_axes=("difficulty_stratum", "length", "genotype_size"),
    description="Per-read runtime, memory, and candidate-count instrumentation.",
)

REGION_LABELS_MANIFEST = ScoringComponentManifest(
    name="region_labels",
    scope="global",
    metric_keys=_region_metric_keys(),
    required_prediction_fields=("region_labels",),
    ground_truth_fields=("region_labels", "presented_region_labels"),
    scenario_axes=("difficulty_stratum", "length", "segment_presence"),
    description="Per-position V/D/J/N/pre/post region labels.",
)

STATE_LABELS_MANIFEST = ScoringComponentManifest(
    name="state_labels",
    scope="global",
    metric_keys=_state_metric_keys(),
    required_prediction_fields=("state_labels",),
    ground_truth_fields=("state_labels", "presented_state_labels"),
    scenario_axes=("mutation_burden", "indel_burden", "noise_burden"),
    description="Per-position germline/substitution/insertion/deletion state labels.",
)

NOISE_COUNT_MANIFEST = ScoringComponentManifest(
    name="noise_count",
    scope="global",
    metric_keys=("noise_count_mae",),
    required_prediction_fields=("noise_count",),
    ground_truth_fields=("noise_count",),
    scenario_axes=("noise_burden",),
    description="Sequencing/PCR noise burden scalar scoring.",
)

MUTATION_RATE_MANIFEST = ScoringComponentManifest(
    name="mutation_rate",
    scope="global",
    metric_keys=("mutation_rate_mae",),
    required_prediction_fields=("mutation_rate",),
    ground_truth_fields=("mutation_rate",),
    scenario_axes=("mutation_burden",),
    description="SHM mutation-rate scalar scoring.",
)

INDEL_COUNT_MANIFEST = ScoringComponentManifest(
    name="indel_count",
    scope="global",
    metric_keys=("indel_count_mae", "indel_length_mae"),
    required_prediction_fields=("indel_count",),
    ground_truth_fields=("indel_count",),
    scenario_axes=("indel_burden",),
    description="Indel-count scalar scoring.",
)

PRODUCTIVE_MANIFEST = ScoringComponentManifest(
    name="productive",
    scope="global",
    metric_keys=("productive_acc",),
    required_prediction_fields=("productive",),
    ground_truth_fields=("productive",),
    scenario_axes=("productivity", "junction_biology"),
    description="Productivity flag scoring.",
)

GENE_MANIFEST = ScoringComponentManifest(
    name="gene",
    scope="gene",
    metric_keys=(
        "found_rate",
        "missing_call_rate",
        "call_top1_in_set",
        "gene_top1_in_set",
        "call_set_precision",
        "call_set_recall",
        "call_set_f1",
        "call_exact_set",
        "truth_set_size",
        "pred_set_size",
        "set_size_mae",
        "overcall_rate",
        "undercall_rate",
        "graceful_abstain",
        "graceful_useful",
        "graceful_hard_error",
        "graceful_non_error",
        *_coordinate_metric_keys(),
        "coordinate_parse_rate",
        "missing_coordinate_rate",
        "off_by_one_rate",
        "coordinate_frame_error_rate",
        "seq_span_iou",
        "segment_length_mae",
        "negative_span_rate",
        "top1_recall",
        "top3_recall",
        "top5_recall",
        "top10_recall",
        "topk_truth_set_recall",
        "same_gene_sibling_top1",
        "sibling_set_recall",
        "outside_genotype_call_rate",
        "genotype_restricted_call_acc",
        "cigar_exact",
        "cigar_edit_distance",
        "gap_event_f1",
        "trim_5_mae",
        "trim_3_mae",
        "identity_mae",
    ),
    required_prediction_fields=("v/d/j calls", "v/d/j sequence/germline coordinates"),
    ground_truth_fields=("v_call", "d_call", "j_call", "v/d/j coordinates", "v/d/j cigar"),
    scenario_axes=("allele_ambiguity", "genotype_size", "segment_presence", "orientation"),
    description="Per-gene call, set, coordinate, top-k, genotype-mask, and record-field scoring.",
)

GLOBAL_SCORING_MANIFESTS: tuple[ScoringComponentManifest, ...] = (
    ORIENTATION_MANIFEST,
    AIRR_CONTRACT_MANIFEST,
    SEGMENT_ORDER_MANIFEST,
    JUNCTION_MANIFEST,
    METADATA_MANIFEST,
    PERFORMANCE_MANIFEST,
    REGION_LABELS_MANIFEST,
    STATE_LABELS_MANIFEST,
    NOISE_COUNT_MANIFEST,
    MUTATION_RATE_MANIFEST,
    INDEL_COUNT_MANIFEST,
    PRODUCTIVE_MANIFEST,
)

SCORING_MANIFESTS: tuple[ScoringComponentManifest, ...] = (*GLOBAL_SCORING_MANIFESTS, GENE_MANIFEST)


def scoring_manifest_catalog(*, include_metric_specs: bool = True) -> list[dict[str, Any]]:
    """Return the scoring component manifest as serializable dictionaries."""

    return [manifest.to_dict(include_metric_specs=include_metric_specs) for manifest in SCORING_MANIFESTS]


def validate_scoring_manifest(
    manifests: tuple[ScoringComponentManifest, ...] = SCORING_MANIFESTS,
) -> dict[str, Any]:
    """Validate scorer manifest uniqueness and registry coverage."""

    from ...core import SCENARIO_AXES, metric_registry

    registry_keys = set(metric_registry())
    axis_names = {axis.name for axis in SCENARIO_AXES}
    names = [manifest.name for manifest in manifests]
    duplicate_names = tuple(sorted({name for name in names if names.count(name) > 1}))
    duplicate_metric_keys = tuple(
        sorted(
            f"{manifest.name}:{key}"
            for manifest in manifests
            for key in set(manifest.metric_keys)
            if manifest.metric_keys.count(key) > 1
        )
    )
    metric_keys = {key for manifest in manifests for key in manifest.metric_keys}
    unregistered = tuple(sorted(metric_keys - registry_keys))
    unknown_axes = tuple(
        sorted(
            f"{manifest.name}:{axis}"
            for manifest in manifests
            for axis in manifest.scenario_axes
            if axis not in axis_names
        )
    )
    problems = []
    if duplicate_names:
        problems.append("duplicate_scoring_component_names")
    if duplicate_metric_keys:
        problems.append("duplicate_metric_keys_within_component")
    if unknown_axes:
        problems.append("unknown_scoring_manifest_axes")
    if unregistered:
        problems.append("scoring_manifest_metric_keys_without_registry")
    problems = tuple(problems)
    return {
        "valid": not problems,
        "problems": problems,
        "n_components": len(manifests),
        "n_metric_keys": len(metric_keys),
        "duplicate_component_names": duplicate_names,
        "duplicate_metric_keys_within_component": duplicate_metric_keys,
        "metric_keys_without_registry": unregistered,
        "unknown_axes": unknown_axes,
    }
