"""Measurement-aligned generation scenario catalog."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from ...core.criteria import CRITERIA

ALLOWED_MEASUREMENT_SCENARIO_STATUSES = ("integrated", "coverage_planned", "planned")


@dataclass(frozen=True)
class MeasurementScenario:
    """A benchmark measurement surface and the GenAIRR slice designed for it."""

    name: str
    status: str
    criteria: tuple[str, ...]
    scenario_axes: tuple[str, ...]
    genairr_features: tuple[str, ...]
    stratum_names: tuple[str, ...] = ()
    required_coverage_labels: tuple[str, ...] = ()
    isolated_variables: tuple[str, ...] = ()
    controlled_variables: tuple[str, ...] = ()
    notes: str = ""

    @property
    def metric_keys(self) -> tuple[str, ...]:
        return _metric_keys_for(self.criteria)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["metric_keys"] = self.metric_keys
        return data


_CRITERIA_BY_NAME = {criterion.name: criterion for criterion in CRITERIA}


def _metric_keys_for(criteria: tuple[str, ...]) -> tuple[str, ...]:
    keys: list[str] = []
    for name in criteria:
        criterion = _CRITERIA_BY_NAME.get(name)
        if criterion is None:
            continue
        keys.extend(criterion.metric_keys)
    return tuple(dict.fromkeys(keys))


MEASUREMENT_SCENARIOS: tuple[MeasurementScenario, ...] = (
    MeasurementScenario(
        name="airr_contract_baseline",
        status="integrated",
        criteria=("airr_schema_completeness", "prediction_completeness"),
        scenario_axes=("difficulty_stratum", "locus_chain"),
        genairr_features=("vdj_recombination", "metadata_stamping"),
        stratum_names=("clean_full", "moderate_full", "hard_full"),
        required_coverage_labels=("stratum:clean_full", "stratum:moderate_full", "stratum:hard_full"),
        isolated_variables=("valid GenAIRR AIRR record contract",),
        controlled_variables=("no forced contamination", "no forced read-layout transform"),
        notes="Baseline records make output-contract failures visible before stress slices are interpreted.",
    ),
    MeasurementScenario(
        name="coordinate_and_orientation",
        status="integrated",
        criteria=("coordinate_convention_compliance", "orientation_detection"),
        scenario_axes=("orientation", "segment_presence"),
        genairr_features=("deterministic_orientation_transforms",),
        stratum_names=("orientation", "all_orientations_hard", "adaptive_fr3_revcomp"),
        required_coverage_labels=(
            "orientation:identity",
            "orientation:reverse_complement",
            "orientation:complement",
            "orientation:reverse",
        ),
        isolated_variables=("presented read orientation",),
        controlled_variables=("canonical GenAIRR truth retained separately from presented truth",),
    ),
    MeasurementScenario(
        name="segmentation_and_boundaries",
        status="integrated",
        criteria=("per_position_region_tags", "in_sequence_boundaries", "segment_length_and_order", "germline_boundaries"),
        scenario_axes=("difficulty_stratum", "length", "segment_presence", "orientation"),
        genairr_features=("vdj_recombination", "end_loss_trimming", "adaptive_anchor_amplicons"),
        stratum_names=("clean_full", "moderate_full", "trimmed", "amplicon_fr1", "amplicon_fr2", "amplicon_janchor", "amplicon_jxshort"),
        required_coverage_labels=("segment_presence:all_segments_visible", "segment_presence:short_v_tail"),
        isolated_variables=("V/D/J region and boundary localization",),
        controlled_variables=("allele truth derived from the same GenAIRR record used for labels",),
    ),
    MeasurementScenario(
        name="junction_and_np_recovery",
        status="integrated",
        criteria=("junction_and_np_regions", "junction_sequence_and_translation", "np_and_p_nucleotide_recovery"),
        scenario_axes=("junction_biology", "length", "segment_presence"),
        genairr_features=("vdj_recombination",),
        stratum_names=("clean_full", "amplicon_janchor", "amplicon_jxshort", "amplicon_janchor_rc", "ultra_short_fragment_40"),
        required_coverage_labels=(
            "junction_length:short_junction",
            "junction_length:typical_junction",
            "junction_length:long_junction",
        ),
        isolated_variables=("junction/CDR3 and N/P-addition annotation",),
        controlled_variables=("short reads are GenAIRR end-loss amplicons (V/FR-anchored or J-anchored), not post-hoc crops",),
    ),
    MeasurementScenario(
        name="shm_isolation",
        status="integrated",
        criteria=(
            "allele_top1_call",
            "mutation_site_detection",
            "regional_shm_profile",
            "mutation_noise_indel_scalars",
            "segment_identity_and_support",
        ),
        scenario_axes=("mutation_burden", "allele_ambiguity"),
        genairr_features=("s5f_mutation", "segment_targeted_mutation"),
        stratum_names=("high_shm", "high_shm_extreme"),
        required_coverage_labels=("mutation:12-18%", "mutation:>18%", "tag:shm"),
        isolated_variables=("somatic hypermutation burden",),
        controlled_variables=("end loss", "indels", "sequencing/PCR noise", "fragment crop"),
    ),
    MeasurementScenario(
        name="indel_isolation",
        status="integrated",
        criteria=("alignment_path_and_cigar", "indel_event_detection", "mutation_noise_indel_scalars"),
        scenario_axes=("indel_burden", "length"),
        genairr_features=("polymerase_indels",),
        stratum_names=("high_indel", "high_indel_extreme"),
        required_coverage_labels=("indel:3-5", "indel:>5", "tag:indel"),
        isolated_variables=("polymerase insertion/deletion burden",),
        controlled_variables=("SHM", "sequencing/PCR noise", "end loss", "fragment crop"),
    ),
    MeasurementScenario(
        name="noise_and_ambiguity_isolation",
        status="integrated",
        criteria=("ambiguous_base_handling", "mutation_noise_indel_scalars", "mutation_site_detection"),
        scenario_axes=("noise_burden", "mutation_burden"),
        genairr_features=("sequencing_errors", "ambiguous_base_calls", "pcr_substitution_errors"),
        stratum_names=("noisy_ambiguous", "ambiguous_n_extreme"),
        required_coverage_labels=("noise:3-8", "noise:>8", "tag:noise", "tag:ambiguous"),
        isolated_variables=("sequencing/PCR error and N-base burden",),
        controlled_variables=("SHM", "indels", "end loss", "fragment crop"),
    ),
    MeasurementScenario(
        name="edit_state_labels",
        status="integrated",
        criteria=("per_position_edit_state",),
        scenario_axes=("mutation_burden", "indel_burden", "noise_burden"),
        genairr_features=("s5f_mutation", "polymerase_indels", "sequencing_errors", "ambiguous_base_calls"),
        stratum_names=("high_shm_extreme", "high_indel_extreme", "ambiguous_n_extreme", "hard_full"),
        required_coverage_labels=("tag:shm", "tag:indel", "tag:noise", "mutation:>18%", "indel:>5", "noise:>8"),
        isolated_variables=("per-position germline/substitution/insertion/deletion state classes",),
        controlled_variables=("single-variable extreme strata isolate each edit-state driver"),
    ),
    MeasurementScenario(
        name="end_loss_and_layout",
        status="integrated",
        criteria=("trim_recovery", "read_layout_and_end_loss"),
        scenario_axes=("read_layout", "segment_presence", "length"),
        genairr_features=("end_loss_trimming", "paired_end_reads"),
        stratum_names=("trimmed", "extreme_end_loss", "paired_end"),
        required_coverage_labels=("tag:trim", "read_layout:paired_end"),
        isolated_variables=("read-end loss", "read layout"),
        controlled_variables=("SHM/noise/indels controlled in focused end-loss and paired-end strata"),
    ),
    MeasurementScenario(
        name="adaptive_fragment_observability",
        status="integrated",
        criteria=(
            "set_valued_allele_call",
            "graceful_degradation",
            "fragment_observability",
            "allele_uncertainty_calibration",
            "boundary_uncertainty_calibration",
        ),
        scenario_axes=("allele_ambiguity", "segment_presence", "length"),
        genairr_features=("adaptive_anchor_amplicons",),
        stratum_names=("adaptive_fr1", "adaptive_fr2", "adaptive_fr3", "adaptive_janchor", "adaptive_fr3_revcomp"),
        required_coverage_labels=("tag:adaptive", "tag:fr1", "tag:fr2", "tag:fr3", "tag:j_anchored"),
        isolated_variables=("observable evidence loss from adaptive-style amplicons",),
        controlled_variables=("GenAIRR full truth retained so scoring can distinguish model error from unobservability"),
    ),
    MeasurementScenario(
        name="d_inversion",
        status="integrated",
        criteria=("d_orientation_and_inversion",),
        scenario_axes=("d_orientation",),
        genairr_features=("d_inversion",),
        stratum_names=("forced_d_inversion",),
        required_coverage_labels=("d_orientation:inverted", "tag:d_inversion"),
        isolated_variables=("D-segment inversion",),
        controlled_variables=("SHM", "indels", "noise", "end loss", "fragment crop"),
    ),
    MeasurementScenario(
        name="receptor_revision",
        status="integrated",
        criteria=("receptor_revision_cases",),
        scenario_axes=("difficulty_stratum",),
        genairr_features=("receptor_revision",),
        stratum_names=("receptor_revision",),
        required_coverage_labels=("revision:yes", "tag:receptor_revision"),
        isolated_variables=("receptor revision-like V replacement",),
        controlled_variables=("SHM", "indels", "noise", "end loss", "fragment crop"),
    ),
    MeasurementScenario(
        name="contaminant_handling",
        status="integrated",
        criteria=("contaminant_and_out_of_scope_handling",),
        scenario_axes=("input_validity",),
        genairr_features=("contamination",),
        stratum_names=("contaminant",),
        required_coverage_labels=("contaminant:yes", "tag:contaminant"),
        isolated_variables=("invalid/contaminant input sequence",),
        controlled_variables=("all receptor-specific corruption is irrelevant after contaminant replacement"),
    ),
    MeasurementScenario(
        name="productivity_and_frame",
        status="coverage_planned",
        criteria=("productive_status", "frame_and_stop_codon"),
        scenario_axes=("productivity", "junction_biology"),
        genairr_features=("productive_only", "vdj_recombination"),
        stratum_names=("productive_only_clean", "clean_full", "hard_full"),
        required_coverage_labels=(
            "productivity:yes",
            "productivity:no",
            "junction_frame:in_frame",
            "junction_frame:out_of_frame",
            "junction_frame:stop_codon",
        ),
        isolated_variables=("productive/frame/stop-codon class",),
        controlled_variables=("productive-only positive control plus coverage-accepted negative classes"),
        notes="GenAIRR has a productive-only constraint; nonproductive subclasses are isolated by coverage acceptance.",
    ),
    MeasurementScenario(
        name="allele_coverage_and_candidates",
        status="coverage_planned",
        criteria=("gene_level_call", "sibling_allele_resolution", "topk_candidate_recall", "rare_allele_coverage"),
        scenario_axes=("allele_frequency", "genotype_size", "allele_ambiguity"),
        genairr_features=("allele_restriction", "vdj_recombination"),
        stratum_names=("clean_full", "moderate_full", "hard_full", "high_shm"),
        required_coverage_labels=(
            "measurement:allele_coverage_and_candidates",
            "ambiguity:any_multi",
            "ambiguity:all_single",
        ),
        isolated_variables=("truth allele identity and candidate recall",),
        controlled_variables=("coverage planning can require per-reference allele and allele/context counts"),
    ),
    MeasurementScenario(
        name="constant_region",
        status="coverage_planned",
        criteria=("constant_region_call",),
        scenario_axes=("read_layout",),
        genairr_features=("vdj_recombination",),
        stratum_names=("clean_full", "moderate_full", "hard_full"),
        required_coverage_labels=("constant_region:present", "constant_region:absent"),
        isolated_variables=("constant-region truth availability",),
        controlled_variables=("current default GenAIRR IGH records often lack C calls; readiness reports this explicitly"),
    ),
    MeasurementScenario(
        name="multi_locus_chain",
        status="coverage_planned",
        criteria=("locus_and_chain_support",),
        scenario_axes=("locus_chain",),
        genairr_features=("multi_locus_configs",),
        required_coverage_labels=("measurement:multi_locus_chain",),
        isolated_variables=("locus and D/no-D chain type",),
        controlled_variables=("same scoring contract across IGH/IGK/IGL/TCR configs"),
    ),
    MeasurementScenario(
        name="genotype_masked_inference",
        status="coverage_planned",
        criteria=("genotype_mask_compliance", "runtime_and_memory"),
        scenario_axes=("genotype_size", "allele_frequency"),
        genairr_features=("genotype_and_cohort_sampling", "allele_restriction"),
        required_coverage_labels=("measurement:genotype_masked_inference",),
        isolated_variables=("candidate genotype size and cohort-specific allele support",),
        controlled_variables=("same query distribution across full-reference and restricted-genotype runs"),
    ),
)


def measurement_scenario_catalog() -> list[dict[str, Any]]:
    """Return measurement-to-generation scenario mappings."""

    return [scenario.to_dict() for scenario in MEASUREMENT_SCENARIOS]


def measurement_scenario_by_name() -> dict[str, MeasurementScenario]:
    """Return measurement scenarios keyed by name."""

    return {scenario.name: scenario for scenario in MEASUREMENT_SCENARIOS}


def case_measurement_scenarios(case, *, statuses: tuple[str, ...] | None = None) -> tuple[str, ...]:
    """Infer which measurement scenarios one generated case contributes to."""

    from ..planner import case_coverage_labels

    allowed_statuses = set(statuses or ALLOWED_MEASUREMENT_SCENARIO_STATUSES)
    labels = set(case_coverage_labels(case))
    record = case.record or {}
    explicit = record.get("benchmark_measurement")
    matched: list[str] = []
    for scenario in MEASUREMENT_SCENARIOS:
        if scenario.status not in allowed_statuses:
            continue
        if explicit == scenario.name:
            matched.append(scenario.name)
            continue
        if case.stratum in scenario.stratum_names:
            matched.append(scenario.name)
            continue
        if labels & set(scenario.required_coverage_labels):
            matched.append(scenario.name)
    return tuple(dict.fromkeys(matched))


def measurement_coverage_summary(cases, *, statuses: tuple[str, ...] | None = None) -> dict[str, Any]:
    """Summarize generated case coverage by intended measurement scenario."""

    from collections import Counter

    case_list = list(cases)
    allowed_statuses = set(statuses or ALLOWED_MEASUREMENT_SCENARIO_STATUSES)
    rows = []
    by_measurement: Counter[str] = Counter()
    explicit_by_measurement: Counter[str] = Counter()
    unmapped = []
    for case in case_list:
        names = case_measurement_scenarios(case, statuses=tuple(allowed_statuses))
        if not names:
            unmapped.append(case.case_id)
        by_measurement.update(names)
        explicit = (case.record or {}).get("benchmark_measurement")
        if explicit:
            explicit_by_measurement[explicit] += 1

    for scenario in MEASUREMENT_SCENARIOS:
        if scenario.status not in allowed_statuses:
            continue
        rows.append(
            {
                "name": scenario.name,
                "status": scenario.status,
                "n_cases": by_measurement.get(scenario.name, 0),
                "n_explicit_cases": explicit_by_measurement.get(scenario.name, 0),
                "criteria": scenario.criteria,
                "metric_keys": scenario.metric_keys,
                "stratum_names": scenario.stratum_names,
                "required_coverage_labels": scenario.required_coverage_labels,
            }
        )

    return {
        "n_cases": len(case_list),
        "n_scenarios": len(rows),
        "by_measurement": dict(sorted(by_measurement.items())),
        "explicit_by_measurement": dict(sorted(explicit_by_measurement.items())),
        "unmapped_case_ids": unmapped[:25],
        "n_unmapped_cases": len(unmapped),
        "scenarios": rows,
    }


def _spec_measurement_names(spec) -> set[str]:
    names = set()
    if spec is None:
        return names
    for stratum in spec.strata:
        metadata = (stratum.param_overrides or {}).get("metadata") or {}
        measurement = metadata.get("benchmark_measurement")
        if measurement:
            names.add(str(measurement))
    return names


def measurement_required_contexts(
    *,
    statuses: tuple[str, ...] = ("integrated", "coverage_planned"),
    spec=None,
) -> tuple[str, ...]:
    """Return concrete coverage labels required by measurement-aligned generation."""

    labels: list[str] = []
    known_strata = {stratum.name for stratum in spec.strata} if spec is not None else None
    explicit_measurements = _spec_measurement_names(spec)
    for scenario in MEASUREMENT_SCENARIOS:
        if scenario.status not in statuses:
            continue
        if known_strata is None:
            labels.extend(label for label in scenario.required_coverage_labels if "{" not in label)
            labels.extend(f"stratum:{name}" for name in scenario.stratum_names)
            continue

        stratum_names = tuple(name for name in scenario.stratum_names if name in known_strata)
        explicit_in_spec = scenario.name in explicit_measurements
        if not stratum_names and not explicit_in_spec:
            continue
        for label in scenario.required_coverage_labels:
            if "{" in label:
                continue
            if label.startswith("measurement:") and not explicit_in_spec:
                continue
            labels.append(label)
        labels.extend(f"stratum:{name}" for name in stratum_names)
    return tuple(dict.fromkeys(labels))


def measurement_aligned_coverage_plan(
    spec,
    reference_set=None,
    *,
    min_cases: int | None = None,
    min_per_measurement_context: int = 1,
    min_per_allele: int = 0,
    min_per_orientation: int = 0,
    min_per_context: int = 0,
    min_per_stratum: int = 0,
    min_per_allele_context: int = 0,
    allele_contexts: tuple[str, ...] | None = None,
    max_candidates: int | None = None,
    name: str = "measurement_aligned",
):
    """Build a coverage plan from measurement-aligned scenario requirements."""

    from ..planner import coverage_plan_from_spec

    required_labels = {
        label: int(min_per_measurement_context)
        for label in measurement_required_contexts(spec=spec)
        if min_per_measurement_context > 0
    }
    return coverage_plan_from_spec(
        spec,
        reference_set,
        min_cases=min_cases,
        min_per_allele=min_per_allele,
        min_per_orientation=min_per_orientation,
        min_per_context=min_per_context,
        min_per_stratum=min_per_stratum,
        min_per_allele_context=min_per_allele_context,
        allele_contexts=allele_contexts,
        required_labels=required_labels,
        max_candidates=max_candidates,
        name=name,
    )


def validate_measurement_scenario_catalog(spec=None) -> dict[str, Any]:
    """Validate measurement scenario references against criteria, metrics, axes, and strata."""

    from ...core import SCENARIO_AXES, metric_registry

    errors: list[str] = []
    names = [scenario.name for scenario in MEASUREMENT_SCENARIOS]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        errors.append(f"duplicate measurement scenarios: {duplicates}")

    criteria_names = set(_CRITERIA_BY_NAME)
    metric_keys = set(metric_registry())
    axis_names = {axis.name for axis in SCENARIO_AXES}
    known_strata = {stratum.name for stratum in spec.strata} if spec is not None else None
    represented_criteria: set[str] = set()
    represented_axes: set[str] = set()

    for scenario in MEASUREMENT_SCENARIOS:
        if scenario.status not in ALLOWED_MEASUREMENT_SCENARIO_STATUSES:
            errors.append(f"{scenario.name}: invalid status {scenario.status!r}")
        unknown_criteria = sorted(set(scenario.criteria) - criteria_names)
        if unknown_criteria:
            errors.append(f"{scenario.name}: unknown criteria {unknown_criteria}")
        unknown_metrics = sorted(set(scenario.metric_keys) - metric_keys)
        if unknown_metrics:
            errors.append(f"{scenario.name}: unknown metric keys {unknown_metrics}")
        unknown_axes = sorted(set(scenario.scenario_axes) - axis_names)
        if unknown_axes:
            errors.append(f"{scenario.name}: unknown scenario axes {unknown_axes}")
        if known_strata is not None and scenario.status != "planned":
            unknown_strata = sorted(set(scenario.stratum_names) - known_strata)
            if unknown_strata:
                errors.append(f"{scenario.name}: strata missing from spec {unknown_strata}")
        if scenario.status != "planned" and not scenario.required_coverage_labels:
            errors.append(f"{scenario.name}: active scenario has no required coverage labels")
        represented_criteria.update(scenario.criteria)
        represented_axes.update(scenario.scenario_axes)

    missing_criteria = sorted(criteria_names - represented_criteria)
    if missing_criteria:
        errors.append(f"criteria without measurement scenarios: {missing_criteria}")

    try:
        from ...evaluation.scoring.manifest import SCORING_MANIFESTS

        manifest_axes = {axis for manifest in SCORING_MANIFESTS for axis in manifest.scenario_axes}
        missing_manifest_axes = sorted(manifest_axes - represented_axes)
    except Exception:
        missing_manifest_axes = []
    if missing_manifest_axes:
        errors.append(f"scoring manifest axes without measurement scenarios: {missing_manifest_axes}")

    return {
        "valid": not errors,
        "errors": errors,
        "n_scenarios": len(MEASUREMENT_SCENARIOS),
        "statuses": dict(
            sorted(
                (status, sum(1 for scenario in MEASUREMENT_SCENARIOS if scenario.status == status))
                for status in ALLOWED_MEASUREMENT_SCENARIO_STATUSES
            )
        ),
        "n_represented_criteria": len(represented_criteria),
        "n_required_contexts": len(measurement_required_contexts()),
    }
