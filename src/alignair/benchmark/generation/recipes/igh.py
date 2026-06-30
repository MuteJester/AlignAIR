"""Default human IGH benchmark recipe composition."""
from __future__ import annotations

from ...core.schema import BenchmarkSpec, StratumSpec
from .adaptive import adaptive_igh_strata
from .background import isolated_params
from .focused import focused_igh_strata


def default_igh_spec(n_per_stratum: int = 200, seed: int = 123) -> BenchmarkSpec:
    """Default human IGH benchmark recipe over ``GenAIRR.data.HUMAN_IGH_OGRDB``."""

    strata = (
        StratumSpec(
            name="clean_full",
            n=n_per_stratum,
            progress=0.0,
            param_overrides=isolated_params(),
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
            progress=0.0,
            param_overrides=isolated_params(),
            orientation_ids=(0, 1, 2, 3),
            description="Clean records cycled through all four orientation transforms.",
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


def default_igh_assay_spec(
    n_per_stratum: int = 200,
    n_per_focus: int = 200,
    n_per_adaptive: int | None = None,
    seed: int = 123,
) -> BenchmarkSpec:
    """Broad, focused, and adaptive human IGH recipe for assay-style evaluations."""

    broad = default_igh_spec(n_per_stratum=n_per_stratum, seed=seed)
    adaptive_n = n_per_focus if n_per_adaptive is None else n_per_adaptive
    return BenchmarkSpec(
        name="human_igh_ogrdb_assay",
        dataconfig_name=broad.dataconfig_name,
        seed=seed,
        strata=broad.strata + focused_igh_strata(n_per_focus) + adaptive_igh_strata(adaptive_n),
        description=(
            "Broad, focused, and adaptive-amplicon GenAIRR human IGH "
            "benchmark for end-to-end AIRR alignment assay reports."
        ),
    )
