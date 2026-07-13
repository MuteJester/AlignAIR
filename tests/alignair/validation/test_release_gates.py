"""P0-16 acceptance: every model-card claim maps to a real named test, thresholds are sane and
version-controlled, and (when the shipped model is present) the model clears the scientific gates."""
import os
import re

import pytest

from alignair.validation.gates import CLAIM_TESTS, SCIENTIFIC_THRESHOLDS

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
_IGH = ".private/models/alignair_igh_v1.cal.alignair"


@pytest.mark.parametrize("claim,nodeid", list(CLAIM_TESTS.items()))
def test_every_claim_points_at_a_real_test(claim, nodeid):
    path, _, func = nodeid.partition("::")
    full = os.path.join(_ROOT, path)
    assert os.path.exists(full), f"claim {claim!r}: test file {path} does not exist"
    src = open(full).read()
    assert re.search(rf"def {re.escape(func)}\b", src), f"claim {claim!r}: {func} not found in {path}"


def test_thresholds_are_sane():
    assert SCIENTIFIC_THRESHOLDS and all(0.0 <= v <= 1.0 for v in SCIENTIFIC_THRESHOLDS.values())


@pytest.mark.slow
@pytest.mark.skipif(not os.path.exists(_IGH), reason="shipped IGH model not present")
def test_igh_model_meets_scientific_gates():
    """The production IGH model clears the version-controlled per-task floors on a fixed seeded set."""
    import itertools

    import GenAIRR.data as gd
    from alignair.api import load_model
    from alignair.core.config import AlignAIRConfig  # noqa: F401 (ensures package import path)
    from alignair.train.gym import Curriculum, build_experiment
    from alignair.train.trainer import build_batch, eval_metrics

    model, ref = load_model(_IGH, device="cpu")
    exp = build_experiment(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.3)), allow_curatable=True)
    recs = [r for r in itertools.islice(exp.stream_records(n=None, seed=7), 400)
            if all(r.get(f"{g}_sequence_start") is not None for g in ("v", "d", "j"))][:128]
    batch_in, targets = build_batch(recs, ref, model.cfg, device="cpu")
    import torch
    with torch.no_grad():
        metrics = eval_metrics(model(batch_in), targets, model.cfg)
    failures = {k: (metrics.get(k), thr) for k, thr in SCIENTIFIC_THRESHOLDS.items()
                if metrics.get(k) is not None and metrics[k] < thr}
    assert not failures, f"model below release gates: {failures}"
