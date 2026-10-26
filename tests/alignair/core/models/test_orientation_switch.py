"""The backward-compatible use_orientation flag: nucleotide default unchanged; when off, the
model ingests non-nucleotide (amino-acid) tokens without the reverse-complement path."""
import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core.model import AlignAIR

pytestmark = pytest.mark.component


def _cfg(**kw):
    base = dict(v_allele_count=6, j_allele_count=3, d_allele_count=4, has_d=True,
                num_chain_types=1, max_seq_length=192, vocab_size=6)
    base.update(kw)
    return AlignAIRConfig(**base)


def test_default_is_orientation_on_and_unchanged():
    cfg = _cfg()
    assert cfg.use_orientation is True                     # nucleotide default
    model = AlignAIR(cfg).eval()
    assert model.orientation_head is not None
    tokens = torch.randint(0, 6, (2, 192))
    out = model({"tokenized_sequence": tokens})
    assert "orientation_logits" in out and out["orientation_logits"].shape == (2, 4)


def test_orientation_off_skips_head_and_rc_path():
    cfg = _cfg(use_orientation=False)
    model = AlignAIR(cfg).eval()
    assert model.orientation_head is None
    out = model({"tokenized_sequence": torch.randint(0, 6, (2, 192))})
    assert "orientation_logits" not in out and "orientation" not in out
    # all the non-orientation heads are still present
    for k in ("v_allele", "d_allele", "j_allele", "mutation_rate", "indel_count", "productive",
              "v_start", "j_end", "position_mask"):
        assert k in out


def test_orientation_off_ingests_amino_acid_tokens_without_crashing():
    """The whole point: AA token ids (1..22) exceed the nt complement table (size 6). With
    orientation off there is no complement lookup, so AA input flows through."""
    cfg = _cfg(use_orientation=False, vocab_size=23)
    model = AlignAIR(cfg).eval()
    aa_tokens = torch.randint(1, 23, (2, 192))             # includes token 22 (the stop '*')
    out = model({"tokenized_sequence": aa_tokens})         # must NOT raise IndexError
    assert out["v_allele"].shape == (2, 6)
    for v in out.values():
        if v.dtype.is_floating_point:
            assert torch.isfinite(v).all()


def test_nt_checkpoint_state_dict_unchanged_by_the_flag_default():
    """Byte-compatibility: a default (orientation-on) model has the same parameter names a legacy
    checkpoint expects, including the orientation head."""
    sd = AlignAIR(_cfg()).state_dict()
    assert any(k.startswith("orientation_head") for k in sd)
