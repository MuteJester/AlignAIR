import pytest
import torch

from alignair.aa.config import aa_igh_config
from alignair.aa.model import AAAlignAIR
from alignair.aa.train import aa_eval_metrics, assert_batch_diverse, overfit_one_batch

pytestmark = pytest.mark.integration


def test_eval_metrics_on_perfect_predictions():
    B = 3
    out = {}
    for g in ("v", "j"):
        lg = torch.full((B, 4), -10.0)
        lg[range(B), 2] = 10.0
        out[f"{g}_class_logits"] = lg
        out[f"{g}_start"] = torch.full((B, 1), 5.0)
        out[f"{g}_end"] = torch.full((B, 1), 40.0)
    tg = {"v_class": torch.full((B,), 2), "j_class": torch.full((B,), 2),
          "v_start": torch.full((B, 1), 5.0), "v_end": torch.full((B, 1), 40.0),
          "j_start": torch.full((B, 1), 5.0), "j_end": torch.full((B, 1), 40.0)}
    m = aa_eval_metrics(out, tg)
    assert m["v_class_acc"] == 1.0 and m["j_class_acc"] == 1.0
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert m[k] < 1e-4


def test_assert_batch_diverse_rejects_degenerate_batches():
    same = [{"v_class": 1, "j_class": 0} for _ in range(8)]
    with pytest.raises(ValueError, match="diver"):
        assert_batch_diverse(same)
    mixed = [{"v_class": i % 4, "j_class": i % 2} for i in range(8)]
    assert assert_batch_diverse(mixed) is None


def test_diverse_selector_prioritizes_unseen_classes():
    from alignair.aa.train import diverse_selector
    sel = diverse_selector(min_v_classes=2, min_j_classes=2)
    chosen = [{"v_class": 1, "j_class": 0}]
    assert sel({"v_class": 2, "j_class": 0}, chosen)          # new V -> take
    assert sel({"v_class": 1, "j_class": 5}, chosen)          # new J -> take
    assert not sel({"v_class": 1, "j_class": 0}, chosen)      # nothing new, quota unmet -> skip


@pytest.mark.slow
def test_overfit_one_batch_learns_classes_and_all_four_boundaries():
    torch.manual_seed(0)
    from alignair.aa.data import build_class_maps, collate_aa, collect_usable
    from alignair.aa.tokenizer import AAProteinTokenizer
    from alignair.aa.train import diverse_selector
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"])
    samples, _ = collect_usable(start_seed=7_000_000, budget=50_000, want=16, class_maps=cmaps,
                                tokenizer=AAProteinTokenizer(), max_len=cfg.max_seq_length,
                                select=diverse_selector())    # actively diverse, not hoped-for
    assert_batch_diverse(samples)                       # the proof must not be trivially degenerate
    bi, tg = collate_aa(samples, cfg.max_seq_length)
    res = overfit_one_batch(AAAlignAIR(cfg), bi, tg, steps=300, lr=1e-3)
    assert res["final_loss"] < 0.5 * res["initial_loss"]
    assert res["v_class_acc"] >= 0.9 and res["j_class_acc"] >= 0.9
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert res[k] < 5.0, k
    # both components must actually fall (a falling total must not hide a stuck class term)
    h = res["history"]
    assert h[-1]["classification"] < 0.5 * h[0]["classification"]
    assert h[-1]["segmentation"] < h[0]["segmentation"]
