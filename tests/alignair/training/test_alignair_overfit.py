"""End-to-end overfit smoke: the faithful AlignAIR trains on a fixed tiny gym batch."""
import itertools

import pytest
import torch

from alignair.core.config import AlignAIRConfig
from alignair.core.losses import make_logvars
from alignair.core import AlignAIR
from alignair.training.alignair_trainer import build_batch, train_step


def _fixed_batch(cfg, ref, n=4):
    import GenAIRR.data as gd
    from alignair.gym import Curriculum, build_experiment
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
    logvars = make_logvars(cfg)
    opt = torch.optim.AdamW(list(model.parameters()) + list(logvars.parameters()), lr=1e-3)

    init, _ = train_step(model, batch_in, targets, cfg, logvars, opt)
    last = init
    for _ in range(120):
        last, parts = train_step(model, batch_in, targets, cfg, logvars, opt)
    assert last < 0.7 * init, f"no learning: init={init:.3f} last={last:.3f}"
