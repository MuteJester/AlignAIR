"""Genotype: threshold calibration harness."""
import pytest


@pytest.mark.slow
def test_calibrate_returns_best_params_and_results():
    import GenAIRR.data as gd
    from alignair.core import AlignAIR
    from alignair.core.config import AlignAIRConfig
    from alignair.genotype.calibrate import build_cases, calibrate
    from alignair.reference.reference_set import ReferenceSet
    cfg = AlignAIRConfig.from_dataconfigs(gd.HUMAN_IGH_OGRDB, max_seq_length=576)
    model, ref = AlignAIR(cfg).eval(), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    cases = build_cases(model, ref, gd.HUMAN_IGH_OGRDB, seeds=(0,), strata=("moderate",),
                        depths=(20,), device="cpu", batch_size=20)
    assert cases and "aligned" in cases[0]
    grid = {"present_thr": [0.002, 0.004], "min_support": [0.003]}     # tiny grid for speed
    best, results = calibrate(model, ref, gd.HUMAN_IGH_OGRDB, grid=grid, cases=cases)
    assert len(results) == 2 and "params" in best
    assert set(best) >= {"precision", "recall", "zygosity_acc", "f1", "objective"}
