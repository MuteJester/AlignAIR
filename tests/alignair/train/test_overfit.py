"""End-to-end overfit smoke: the faithful AlignAIR trains on a fixed tiny gym batch."""
import itertools

import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core import AlignAIR
from alignair.train.trainer import build_batch
from alignair.train import overfit_one_batch
from alignair.data.sampling import ClassDiversity
from alignair.core import prepare_model

pytestmark = pytest.mark.integration


def _fixed_batch(cfg, ref, n=4):
    import GenAIRR.data as gd
    from alignair.train.gym import Curriculum, build_experiment
    p = dict(Curriculum().params(0.3))
    p["invert_d_prob"] = 0.0
    exp = build_experiment(gd.HUMAN_IGH_OGRDB, p, allow_curatable=True)
    recs = [r for r in itertools.islice(exp.stream_records(n=None, seed=0), 60)
            if all(r.get(f"{g}_sequence_start") is not None for g in ("v", "d", "j"))][:n]
    assert len(recs) == n
    return build_batch(recs, ref, cfg)


@pytest.mark.slow
def test_overfit_tiny_batch_learns():
    from alignair.reference.reference_set import ReferenceSet
    import GenAIRR.data as gd
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cfg = AlignAIRConfig(max_seq_length=576, has_d=True,
                         v_allele_count=len(ref.gene("V")), j_allele_count=len(ref.gene("J")),
                         d_allele_count=len(ref.gene("D")))
    batch_in, targets = _fixed_batch(cfg, ref)

    torch.manual_seed(0)
    model = AlignAIR(cfg)
    result = overfit_one_batch(model, batch_in, targets, steps=121, lr=1e-3)
    assert result["final_loss"] < 0.7 * result["initial_loss"]
    assert len(result["history"]) == 121
    assert {"v_allele_top1", "d_allele_top1", "j_allele_top1"} <= result.keys()
    assert not model.training


@pytest.mark.slow
def test_overfit_one_batch_learns_classes_and_all_four_boundaries():
    torch.manual_seed(0)
    from alignair.data.adapters import collate_categorical as collate_aa
    from alignair.data.sampling import collect_usable
    from alignair.reference import build_protein_reference
    from alignair.data.tokenizer import AAProteinTokenizer
    diversity = ClassDiversity({"v_class": 3, "j_class": 2})
    setup = prepare_model(build_protein_reference('HUMAN_IGH_OGRDB'), input_space='protein',
                          label_space='protein_classes', has_d=False)
    cmaps, cfg = setup.protein_reference.class_maps, setup.config
    samples, _ = collect_usable(start_seed=7_000_000, budget=50_000, want=16, class_maps=cmaps,
                                tokenizer=AAProteinTokenizer(), max_len=cfg.max_seq_length,
                                select=diversity)    # actively diverse, not hoped-for
    diversity.validate(samples)                       # the proof must not be trivially degenerate
    bi, tg = collate_aa(samples, cfg.max_seq_length)
    res = overfit_one_batch(setup.build_model(), bi, tg, steps=300, lr=1e-3, genes=("v", "j"))
    assert res["final_loss"] < 0.5 * res["initial_loss"]
    assert res["v_class_acc"] >= 0.9 and res["j_class_acc"] >= 0.9
    for k in ("v_start_mae", "v_end_mae", "j_start_mae", "j_end_mae"):
        assert res[k] < 5.0, k
    # both components must actually fall (a falling total must not hide a stuck class term)
    h = res["history"]
    assert h[-1]["classification"] < 0.5 * h[0]["classification"]
    assert h[-1]["segmentation"] < h[0]["segmentation"]
