"""Adaptive and short-amplicon benchmark strata."""
from __future__ import annotations

from ...core.schema import StratumSpec
from .background import isolated_params


def adaptive_igh_strata(n_per_scenario: int = 200) -> tuple[StratumSpec, ...]:
    """Return adaptive/immunoSEQ-style short-read strata."""

    base = dict(progress=0.0, param_overrides=isolated_params())
    return (
        StratumSpec(
            name="adaptive_fr1",
            n=n_per_scenario,
            anchor=("v_germline", 10),
            description="FR1 multiplex-primer amplicon (most 5' V retained).",
            tags=("adaptive", "fr1", "short"),
            **base,
        ),
        StratumSpec(
            name="adaptive_fr2",
            n=n_per_scenario,
            anchor=("v_germline", 80),
            description="FR2 multiplex-primer amplicon.",
            tags=("adaptive", "fr2", "short"),
            **base,
        ),
        StratumSpec(
            name="adaptive_fr3",
            n=n_per_scenario,
            anchor=("v_germline", 200),
            description="FR3 multiplex-primer amplicon (CDR3-proximal, little 5' V).",
            tags=("adaptive", "fr3", "short"),
            **base,
        ),
        StratumSpec(
            name="adaptive_janchor",
            n=n_per_scenario,
            anchor=("j", 110),
            description="J-anchored ~110bp amplicon (3'/J-primer protocols).",
            tags=("adaptive", "j_anchored", "short"),
            **base,
        ),
        StratumSpec(
            name="adaptive_fr3_revcomp",
            n=n_per_scenario,
            anchor=("v_germline", 200),
            orientation_ids=(1,),
            description="Reverse-complement FR3 amplicon (orientation x short product cell).",
            tags=("adaptive", "fr3", "short", "orientation"),
            **base,
        ),
    )
