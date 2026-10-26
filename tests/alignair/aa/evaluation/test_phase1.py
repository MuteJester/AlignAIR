import pytest

pytestmark = pytest.mark.integration


def test_seed_ranges_are_provably_disjoint():
    """The rev-2 defect: replicate streams overlapped ~99%. Every allocated range must be disjoint."""
    from alignair.aa.phase1 import SEED_RANGES, _range_for
    spans = [(_range_for("heldout")["start_seed"], _range_for("heldout")["budget"])]
    for rep in range(8):
        for kind in ("overfit", "train"):
            r = _range_for(kind, rep)
            spans.append((r["start_seed"], r["budget"]))
    spans.sort()
    for (s1, b1), (s2, _) in zip(spans, spans[1:]):
        assert s1 + b1 <= s2, f"seed ranges overlap: [{s1},{s1 + b1}) vs start {s2}"
    # a 3000x64 run needs ~1.4M raw records at ~14% yield; the train budget must exceed that.
    assert SEED_RANGES["train"]["budget"] > 3_000_000


def test_heldout_fingerprint_is_stable_and_content_sensitive():
    import numpy as np
    from alignair.aa.phase1 import heldout_fingerprint

    def s(v=0, vs=0):
        return {"tokens": np.array([1, 2, 3]), "v_class": v, "j_class": 1,
                "v_start": vs, "v_end": 2, "j_start": 2, "j_end": 3}

    base = [s(), s(1)]
    assert heldout_fingerprint(base) == heldout_fingerprint([s(), s(1)])
    assert heldout_fingerprint(base) != heldout_fingerprint([s(), s(2)])      # class change
    assert heldout_fingerprint(base) != heldout_fingerprint([s(0, 1), s(1)])  # boundary change


@pytest.mark.slow
def test_run_learning_proof_smoke_has_provenance_paired_stats_and_disjoint_ranges():
    from alignair.aa.phase1 import run_learning_proof
    r = run_learning_proof(seeds=(0, 1), heldout_size=256, train_steps=10, batch_size=16,
                           overfit_batch=8, overfit_steps=40, device="cpu")
    p = r["provenance"]
    for key in ("class_maps", "class_map_sha256", "db_hash", "config", "versions",
                "git_commit", "param_count"):
        assert key in p
    assert set(p["class_maps"]) == {"V", "J"} and set(p["class_map_sha256"]) == {"V", "J"}
    assert p["versions"]["torch"] and p["versions"]["genairr"]
    assert r["heldout_size"] == 256 and len(r["heldout_fingerprint"]) == 64
    assert set(r["seeds"]) == {"0", "1"}

    one = r["seeds"]["0"]
    assert one["overfit"]["final_loss"] < one["overfit"]["initial_loss"]
    ev = one["heldout"]
    for g in ("v", "j"):
        c = ev["classification"][g]
        assert 0.0 <= c["micro_acc"] <= 1.0 and "majority_baseline" in c
        assert set(c["paired_lift"]) == {"mean", "ci95", "lower_gt_zero"}    # paired gate statistic
    for key in ("v_start", "v_end", "j_start", "j_end"):
        b = ev["boundaries"][key]
        assert "constant_median_baseline_mae" in b and "relative_reduction" in b
        assert set(b["paired_improvement"]) == {"mean", "ci95", "lower_gt_zero"}

    # per-stream accounting, and replicates must occupy DIFFERENT seed ranges
    for kind in ("overfit", "train"):
        c = one["counters"][kind]
        assert c["generated"] == c["usable"] + sum(v for k, v in c.items()
                                                   if k not in ("generated", "usable"))
    a = r["seeds"]["0"]["seed_ranges"]["train"]
    b = r["seeds"]["1"]["seed_ranges"]["train"]
    assert a["last_seed"] < b["first_seed"] or b["last_seed"] < a["first_seed"]
