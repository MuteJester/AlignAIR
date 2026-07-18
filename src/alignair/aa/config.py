"""AA model config factory (Phase 1: V/J only).

Classification heads are sized by PROTEIN-CLASS counts, not raw allele counts. ``has_d=False``:
an unsupervised D branch would drift its BatchNorm running statistics while its weights stay at
init, and emit meaningless outputs. Phase 1b builds a fresh V/D/J config and explicitly transfers
the compatible V/J weights."""
from __future__ import annotations

from ..core.config import AlignAIRConfig
from .tokenizer import AA_VOCAB_SIZE


def aa_igh_config(n_v_classes: int, n_j_classes: int, max_seq_length: int = 192) -> AlignAIRConfig:
    return AlignAIRConfig(
        v_allele_count=n_v_classes,
        j_allele_count=n_j_classes,
        d_allele_count=0,
        has_d=False,
        num_chain_types=1,
        max_seq_length=max_seq_length,
        vocab_size=AA_VOCAB_SIZE,
    )
