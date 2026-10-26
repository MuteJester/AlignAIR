"""Acceptance: every model-card claim maps to a real named test, thresholds are sane and
version-controlled, and (when the shipped model is present) the model clears the scientific gates."""
import ast
import math

import pytest

from alignair.validation.gates import CLAIM_TESTS, SCIENTIFIC_THRESHOLDS

pytestmark = pytest.mark.acceptance


@pytest.mark.parametrize("claim,nodeid", list(CLAIM_TESTS.items()))
def test_every_claim_points_at_a_real_test(claim, nodeid, repo_root):
    path, _, func = nodeid.partition("::")
    full = repo_root / path
    assert full.is_file(), f"claim {claim!r}: test file {path} does not exist"
    src = full.read_text()
    functions = {node.name for node in ast.walk(ast.parse(src)) if isinstance(node, ast.FunctionDef)}
    assert func in functions, f"claim {claim!r}: {func} not found in {path}"


def test_thresholds_are_sane():
    assert SCIENTIFIC_THRESHOLDS and all(0.0 <= v <= 1.0 for v in SCIENTIFIC_THRESHOLDS.values())


@pytest.mark.slow
@pytest.mark.model_validation
def test_igh_model_meets_scientific_gates(igh_model):
    """The production IGH model clears the version-controlled per-task floors on a fixed seeded set."""
    import itertools

    import GenAIRR.data as gd
    from alignair.api import load_model
    from alignair.core.config import AlignAIRConfig  # noqa: F401 (ensures package import path)
    from alignair.train.gym import Curriculum, build_experiment
    from alignair.train.trainer import build_batch, eval_metrics
    from alignair.reference.reference_set import ReferenceSet
    from alignair.model_file.serialize import allele_order_sha256, reference_fasta_sha256
    import torch

    model, ref = load_model(igh_model, device="cpu")
    simulator_reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    assert allele_order_sha256(ref) == allele_order_sha256(simulator_reference)
    assert reference_fasta_sha256(ref) == reference_fasta_sha256(simulator_reference)
    exp = build_experiment(gd.HUMAN_IGH_OGRDB, dict(Curriculum().params(0.3)), allow_curatable=True)
    recs = [r for r in itertools.islice(exp.stream_records(n=None, seed=7), 400)
            if all(r.get(f"{g}_sequence_start") is not None and r.get(f"{g}_sequence_end") is not None
                   and r[f"{g}_sequence_start"] < r[f"{g}_sequence_end"]
                   for g in ("v", "d", "j"))][:128]
    assert len(recs) == 128, "scientific gate requires its complete evaluation batch"
    torch.manual_seed(7)
    batch_in, targets = build_batch(recs, ref, model.cfg, device="cpu")
    with torch.no_grad():
        metrics = eval_metrics(model(batch_in), targets, model.cfg)
    missing = set(SCIENTIFIC_THRESHOLDS) - metrics.keys()
    assert not missing, f"release metrics missing: {sorted(missing)}"
    invalid = {key: metrics[key] for key in SCIENTIFIC_THRESHOLDS
               if metrics[key] is None or not math.isfinite(metrics[key])}
    assert not invalid, f"release metrics must be finite: {invalid}"
    failures = {k: (metrics.get(k), thr) for k, thr in SCIENTIFIC_THRESHOLDS.items()
                if metrics[k] < thr}
    assert not failures, f"model below release gates: {failures}"
