"""Default benchmark strata.

The defaults are intentionally broad rather than clever: they cover clean reads,
moderate/hard corruption, fragments, high indel/error/N regimes, trimming stress,
and orientation stress. Larger public benchmark builds should increase
``n_per_stratum`` and inspect coverage summaries rather than changing the metric.
"""
from __future__ import annotations

from ..core.schema import BenchmarkSpec, StratumSpec


def focused_igh_strata(n_per_scenario: int = 200) -> tuple[StratumSpec, ...]:
    """Focused IGH stress strata for hard-to-hit benchmark labels."""

    return (
        StratumSpec(
            name="productive_only_clean",
            n=n_per_scenario,
            progress=0.0,
            param_overrides={"productive_only": True, "crop_prob": 0.0},
            description="Productive-only near-naive full reads.",
            tags=("full", "clean", "productive"),
        ),
        StratumSpec(
            name="forced_d_inversion",
            n=n_per_scenario,
            progress=0.4,
            param_overrides={"invert_d_prob": 1.0, "crop_prob": 0.0},
            description="Heavy-chain records with D inversion forced.",
            tags=("full", "d_inversion"),
        ),
        StratumSpec(
            name="receptor_revision",
            n=n_per_scenario,
            progress=0.5,
            param_overrides={
                "receptor_revision_prob": 1.0,
                "receptor_revision_same_haplotype": True,
                "crop_prob": 0.0,
            },
            description="Records with GenAIRR receptor-revision events forced.",
            tags=("full", "receptor_revision"),
        ),
        StratumSpec(
            name="contaminant",
            n=n_per_scenario,
            progress=0.4,
            param_overrides={"contaminate_prob": 1.0, "crop_prob": 0.0},
            description="Contaminant-replacement records that should trigger low-confidence/no-call behavior.",
            tags=("full", "contaminant"),
        ),
        StratumSpec(
            name="paired_end",
            n=n_per_scenario,
            progress=0.0,
            param_overrides={
                "crop_prob": 0.0,
                "paired_end": {"r1_length": 150, "r2_length": 150, "insert_size": (250, 420)},
            },
            description="Clean paired-end read-layout records with R1/R2 fields populated.",
            tags=("full", "paired_end", "read_layout"),
        ),
        StratumSpec(
            name="extreme_end_loss",
            n=n_per_scenario,
            progress=0.8,
            param_overrides={
                "end_loss_5": (35, 90),
                "end_loss_3": (35, 90),
                "crop_prob": 0.0,
            },
            description="Severe 5-prime/3-prime end-loss stress.",
            tags=("full", "trim", "end_loss"),
        ),
        StratumSpec(
            name="ultra_short_fragment_40",
            n=n_per_scenario,
            progress=1.0,
            crop_to=40,
            description="Very short junction-centered fragment below the broad benchmark minimum.",
            tags=("fragment", "extreme", "short"),
        ),
        StratumSpec(
            name="high_shm_extreme",
            n=n_per_scenario,
            progress=1.0,
            param_overrides={"mutation_rate": 0.25, "crop_prob": 0.0},
            description="Extreme SHM stress with full reads retained.",
            tags=("full", "shm", "extreme"),
        ),
        StratumSpec(
            name="high_indel_extreme",
            n=n_per_scenario,
            progress=1.0,
            param_overrides={"indel_count": (6, 12), "crop_prob": 0.0},
            description="Extreme polymerase indel stress.",
            tags=("full", "indel", "extreme"),
        ),
        StratumSpec(
            name="ambiguous_n_extreme",
            n=n_per_scenario,
            progress=0.8,
            param_overrides={
                "seq_error_rate": 0.06,
                "ambiguous_count": (12, 30),
                "crop_prob": 0.0,
            },
            description="Heavy ambiguous-base and sequencing-error stress.",
            tags=("full", "noise", "ambiguous", "extreme"),
        ),
        StratumSpec(
            name="all_orientations_hard",
            n=n_per_scenario,
            progress=0.9,
            orientation_ids=(0, 1, 2, 3),
            description="Hard records cycled through every orientation transform.",
            tags=("orientation", "hard"),
        ),
    )


def adaptive_igh_strata(n_per_scenario: int = 200) -> tuple[StratumSpec, ...]:
    """Adaptive/immunoSEQ-style short reads: multiplex V-framework-primer (FR1/FR2/FR3) -> J-primer
    amplicons, modeled as one-sided germline-anchored crops (5' V truncated at the primer site).
    NOTE: raw allele accuracy here is a FLOOR until the coverage-conditioned observable-truth metric
    lands (short windows make many alleles observationally identical)."""
    base = dict(progress=0.8, param_overrides={"crop_prob": 0.0})
    return (
        StratumSpec(name="adaptive_fr1", n=n_per_scenario, anchor=("v_germline", 10),
                    description="FR1 multiplex-primer amplicon (most 5' V retained).",
                    tags=("adaptive", "fr1", "short"), **base),
        StratumSpec(name="adaptive_fr2", n=n_per_scenario, anchor=("v_germline", 80),
                    description="FR2 multiplex-primer amplicon.", tags=("adaptive", "fr2", "short"), **base),
        StratumSpec(name="adaptive_fr3", n=n_per_scenario, anchor=("v_germline", 200),
                    description="FR3 multiplex-primer amplicon (CDR3-proximal, little 5' V).",
                    tags=("adaptive", "fr3", "short"), **base),
        StratumSpec(name="adaptive_janchor", n=n_per_scenario, anchor=("j", 110),
                    description="J-anchored ~110bp amplicon (3'/J-primer protocols).",
                    tags=("adaptive", "j_anchored", "short"), **base),
        StratumSpec(name="adaptive_fr3_revcomp", n=n_per_scenario, anchor=("v_germline", 200),
                    orientation_ids=(1,),
                    description="Reverse-complement FR3 amplicon (orientation x short product cell).",
                    tags=("adaptive", "fr3", "short", "orientation"), **base),
    )


def default_igh_spec(n_per_stratum: int = 200, seed: int = 123) -> BenchmarkSpec:
    """Default human IGH benchmark recipe over ``GenAIRR.data.HUMAN_IGH_OGRDB``."""

    strata = (
        StratumSpec(
            name="clean_full",
            n=n_per_stratum,
            progress=0.0,
            description="Near-naive full-length reads with minimal corruption.",
            tags=("full", "clean"),
        ),
        StratumSpec(
            name="moderate_full",
            n=n_per_stratum,
            progress=0.5,
            description="Moderate SHM/trimming/noise without forced crop.",
            tags=("full", "moderate"),
        ),
        StratumSpec(
            name="hard_full",
            n=n_per_stratum,
            progress=1.0,
            description="Hard full-read corruption regime.",
            tags=("full", "hard"),
        ),
        StratumSpec(
            name="fragment_120",
            n=n_per_stratum,
            progress=1.0,
            crop_to=120,
            description="Junction-centered hard fragment around 120 bp.",
            tags=("fragment", "hard"),
        ),
        StratumSpec(
            name="fragment_80",
            n=n_per_stratum,
            progress=1.0,
            crop_to=80,
            description="Junction-centered hard fragment around 80 bp.",
            tags=("fragment", "hard"),
        ),
        StratumSpec(
            name="fragment_50",
            n=n_per_stratum,
            progress=1.0,
            crop_to=50,
            description="Near-minimal CDR3+flanks hard fragment.",
            tags=("fragment", "extreme"),
        ),
        StratumSpec(
            name="high_shm",
            n=n_per_stratum,
            progress=1.0,
            param_overrides={"mutation_rate": 0.20, "crop_prob": 0.0},
            description="High SHM stress, full reads retained.",
            tags=("full", "shm"),
        ),
        StratumSpec(
            name="high_indel",
            n=n_per_stratum,
            progress=0.8,
            param_overrides={"indel_count": (3, 8), "crop_prob": 0.0},
            description="Polymerase indel stress.",
            tags=("full", "indel"),
        ),
        StratumSpec(
            name="noisy_ambiguous",
            n=n_per_stratum,
            progress=0.8,
            param_overrides={
                "seq_error_rate": 0.04,
                "ambiguous_count": (4, 12),
                "crop_prob": 0.0,
            },
            description="Sequencing-error and ambiguous-base stress.",
            tags=("full", "noise", "ambiguous"),
        ),
        StratumSpec(
            name="trimmed",
            n=n_per_stratum,
            progress=0.8,
            param_overrides={"end_loss_5": (12, 40), "end_loss_3": (12, 40), "crop_prob": 0.0},
            description="5-prime/3-prime end-loss stress.",
            tags=("full", "trim"),
        ),
        StratumSpec(
            name="orientation",
            n=n_per_stratum,
            progress=0.7,
            orientation_ids=(0, 1, 2, 3),
            description="All four orientation transforms, cycled deterministically.",
            tags=("orientation",),
        ),
    )
    return BenchmarkSpec(
        name="human_igh_ogrdb_broad",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=seed,
        strata=strata,
        description="Broad GenAIRR human IGH benchmark for end-to-end alignment tools.",
    )


def focused_igh_spec(n_per_scenario: int = 200, seed: int = 123) -> BenchmarkSpec:
    """Focused human IGH benchmark recipe for hard-to-hit scenario labels."""

    return BenchmarkSpec(
        name="human_igh_ogrdb_focused",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=seed,
        strata=focused_igh_strata(n_per_scenario),
        description="Focused GenAIRR human IGH benchmark for scenario coverage stress tests.",
    )


def default_igh_assay_spec(
    n_per_stratum: int = 200,
    n_per_focus: int = 200,
    seed: int = 123,
) -> BenchmarkSpec:
    """Broad plus focused human IGH benchmark recipe for assay-style evaluations."""

    broad = default_igh_spec(n_per_stratum=n_per_stratum, seed=seed)
    return BenchmarkSpec(
        name="human_igh_ogrdb_assay",
        dataconfig_name=broad.dataconfig_name,
        seed=seed,
        strata=broad.strata + focused_igh_strata(n_per_focus),
        description=(
            "Broad and focused GenAIRR human IGH benchmark for end-to-end "
            "AIRR alignment assay reports."
        ),
    )
