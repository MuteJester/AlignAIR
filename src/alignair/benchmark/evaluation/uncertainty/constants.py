from __future__ import annotations

from ...core.schema import GENES

DEFAULT_BOOTSTRAP_METRICS: tuple[str, ...] = tuple(
    [f"genes.{gene}.{metric}" for gene in GENES for metric in (
        "call_top1_in_set",
        "call_set_f1",
        "ss_mae",
        "se_mae",
        "gs_mae",
        "ge_mae",
    )]
    + [
        "global.junction_nt_exact",
        "global.junction_aa_exact",
        "global.productive_acc",
        "global.orientation_acc",
        "global.required_field_presence",
        "global.parseable_airr_rate",
    ]
)
