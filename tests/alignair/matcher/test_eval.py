
import pytest
import GenAIRR.data as gd
from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch

from alignair.matcher.evaluate import evaluate

pytestmark = pytest.mark.integration


def test_evaluate_returns_per_gene_accuracy_keys():
    cfg = MatcherConfig(d_model=32, d_match=16, n_layers=4, dilations=(1, 2))
    model = AIRRMatch(cfg).eval()
    acc = evaluate(model, gd.HUMAN_IGH_OGRDB, cfg, n=8, progress=0.0, seed=3, device="cpu")
    assert set(acc) == {"V", "D", "J"}
    assert all(0.0 <= v <= 1.0 for v in acc.values())
