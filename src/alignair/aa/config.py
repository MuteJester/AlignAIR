"""AA model config factory.

V/J classification heads are sized by PROTEIN-CLASS counts. The D head (when ``has_d=True``) is sized
by ``n_d_alleles`` — D has no clean protein-equivalence class (it appears in a junction-derived frame,
is trimmed both ends, and absorbs N-nucleotides), so D is predicted at the allele level and its
uncertainty is carried by a calibrated allele set rather than a confident single call."""
from __future__ import annotations

from ..core.config import AlignAIRConfig
from .tokenizer import AA_VOCAB_SIZE


def aa_igh_config(n_v_classes: int, n_j_classes: int, n_d_alleles: int = 0,
                  max_seq_length: int = 192, has_d: bool = False) -> AlignAIRConfig:
    return AlignAIRConfig(
        v_allele_count=n_v_classes,
        j_allele_count=n_j_classes,
        d_allele_count=(n_d_alleles if has_d else 0),
        has_d=has_d,
        num_chain_types=1,
        max_seq_length=max_seq_length,
        vocab_size=AA_VOCAB_SIZE,
    )
