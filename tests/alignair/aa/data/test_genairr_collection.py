"""Real simulator collection budgets and rejection accounting."""
from collections import Counter
import pytest
from alignair.aa.tokenizer import AAProteinTokenizer

pytestmark = pytest.mark.integration


@pytest.mark.slow
def test_collect_usable_real_igh_is_valid_budgeted_and_fully_accounted():
    from alignair.aa.data import build_class_maps, check_counter, collect_usable
    cmaps, meta = build_class_maps("HUMAN_IGH_OGRDB")
    assert meta["n_v_classes"] == 149 and meta["n_j_classes"] == 7
    counter = Counter()
    got, info = collect_usable(start_seed=5_000_000, budget=20_000, want=64, class_maps=cmaps,
                               tokenizer=AAProteinTokenizer(), max_len=192, counter=counter)
    assert len(got) == 64
    check_counter(counter)                                   # generated == usable + rejections
    assert info["first_seed"] == 5_000_000
    assert info["last_seed"] == 5_000_000 + info["generated"] - 1
    for s in got:
        assert 0 <= s["v_class"] < 149 and 0 <= s["j_class"] < 7
        assert 0 <= s["v_start"] < s["v_end"] <= len(s["tokens"])
        assert 0 <= s["j_start"] < s["j_end"] <= len(s["tokens"])
        assert len(s["tokens"]) < 192


@pytest.mark.slow
def test_collect_usable_raises_when_the_seed_budget_is_exhausted():
    from alignair.aa.data import SeedBudgetExhausted, build_class_maps, collect_usable
    cmaps, _ = build_class_maps("HUMAN_IGH_OGRDB")
    with pytest.raises(SeedBudgetExhausted):                  # ~14% yield: 50 records cannot give 500
        collect_usable(start_seed=6_000_000, budget=50, want=500, class_maps=cmaps,
                       tokenizer=AAProteinTokenizer(), max_len=192)
