"""Focused IGH stress-test benchmark recipes."""
from __future__ import annotations

from ...core.schema import BenchmarkSpec, StratumSpec
from .background import isolated_params


def focused_igh_strata(n_per_scenario: int = 200) -> tuple[StratumSpec, ...]:
    """Focused IGH stress strata for hard-to-hit benchmark labels."""

    return (
        StratumSpec(
            name="productive_only_clean",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(productive_only=True),
            description="Productive-only near-naive full reads.",
            tags=("full", "clean", "productive"),
        ),
        StratumSpec(
            name="forced_d_inversion",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(invert_d_prob=1.0),
            description="Heavy-chain records with D inversion forced and background corruption controlled.",
            tags=("full", "d_inversion"),
        ),
        StratumSpec(
            name="receptor_revision",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(
                receptor_revision_prob=1.0,
                receptor_revision_same_haplotype=True,
            ),
            description="Records with GenAIRR receptor-revision events forced and background corruption controlled.",
            tags=("full", "receptor_revision"),
        ),
        StratumSpec(
            name="contaminant",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(contaminate_prob=1.0),
            description="Contaminant-replacement records that should trigger low-confidence/no-call behavior.",
            tags=("full", "contaminant"),
        ),
        StratumSpec(
            name="paired_end",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(
                paired_end={"r1_length": 150, "r2_length": 150, "insert_size": (250, 420)},
            ),
            description="Clean paired-end read-layout records with R1/R2 fields populated.",
            tags=("full", "paired_end", "read_layout"),
        ),
        StratumSpec(
            name="extreme_end_loss",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(
                end_loss_5=(35, 90),
                end_loss_3=(35, 90),
            ),
            description="Severe 5-prime/3-prime end-loss stress.",
            tags=("full", "trim", "end_loss"),
        ),
        StratumSpec(
            name="ultra_short_fragment_40",
            n=n_per_scenario,
            progress=0.0,
            crop_to=40,
            param_overrides=isolated_params(),
            description="Very short junction-centered fragment below the broad benchmark minimum.",
            tags=("fragment", "extreme", "short"),
        ),
        StratumSpec(
            name="high_shm_extreme",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(mutation_rate=0.25),
            description="Extreme SHM stress with full reads retained.",
            tags=("full", "shm", "extreme"),
        ),
        StratumSpec(
            name="high_indel_extreme",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(indel_count=(6, 12)),
            description="Extreme polymerase indel stress.",
            tags=("full", "indel", "extreme"),
        ),
        StratumSpec(
            name="ambiguous_n_extreme",
            n=n_per_scenario,
            progress=0.0,
            param_overrides=isolated_params(
                seq_error_rate=0.06,
                ambiguous_count=(12, 30),
            ),
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


def focused_igh_spec(n_per_scenario: int = 200, seed: int = 123) -> BenchmarkSpec:
    """Focused human IGH benchmark recipe for hard-to-hit scenario labels."""

    return BenchmarkSpec(
        name="human_igh_ogrdb_focused",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=seed,
        strata=focused_igh_strata(n_per_scenario),
        description="Focused GenAIRR human IGH benchmark for scenario coverage stress tests.",
    )
