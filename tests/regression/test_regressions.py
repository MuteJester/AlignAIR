"""
Regression tests for AlignAIR model reproducibility.

These tests compare a fresh snapshot (created from the current code) against
a stored baseline snapshot. They detect unintended changes in:
- Raw model prediction outputs
- Latent space representations
- Pipeline CSV output quality
- Computed quality metrics

Usage:
    pytest tests/regression/ -v
    pytest tests/regression/ -v -k "predictions"
    pytest tests/regression/ -v -k "latent"

Prerequisites:
    - A baseline snapshot must exist at tests/data/snapshots/<model_name>_baseline/
    - The corresponding model bundle must be available in checkpoints/
    - Create a baseline with:
        python -m AlignAIR.Benchmarking snapshot \\
            --model-dir checkpoints/IGH_S5F_576_Extended \\
            --eval-data tests/data/test/sample_igh_extended.csv \\
            --output-dir tests/data/snapshots/IGH_S5F_576_Extended_baseline
"""

import numpy as np
import pytest
from pathlib import Path

pytestmark = [pytest.mark.regression]

REPO_ROOT = Path(__file__).resolve().parents[2]
SNAPSHOTS_DIR = REPO_ROOT / "tests" / "data" / "snapshots"
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"
TEST_DATA_DIR = REPO_ROOT / "tests" / "data" / "test"


def _find_baseline(model_name: str) -> Path:
    """Find a baseline snapshot for the given model."""
    candidate = SNAPSHOTS_DIR / f"{model_name}_baseline"
    if candidate.is_dir() and (candidate / "metadata.json").exists():
        return candidate
    return None


def _find_model_bundle(model_name: str) -> Path:
    """Find a model bundle directory."""
    candidate = CHECKPOINTS_DIR / model_name
    if candidate.is_dir() and (candidate / "config.json").exists():
        return candidate
    # Also check .private/hf_upload/
    private = REPO_ROOT / ".private" / "hf_upload" / model_name
    if private.is_dir() and (private / "config.json").exists():
        return private
    return None


def _find_eval_data(model_name: str) -> Path:
    """Find evaluation data for the given model."""
    # Map model names to test data files
    data_map = {
        "IGH_S5F_576_Extended": "sample_igh_extended.csv",
        "IGH_S5F_576": "sample_igh.csv",
        "IGL_S5F_576": "sample_igl_k.csv",
        "TCRB_UNIFORM_576": "sample_tcrb.csv",
    }
    filename = data_map.get(model_name)
    if filename:
        path = TEST_DATA_DIR / filename
        if path.exists():
            return path
    return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def baseline_snapshot():
    """Load the stored baseline snapshot for IGH_S5F_576_Extended."""
    baseline_path = _find_baseline("IGH_S5F_576_Extended")
    if baseline_path is None:
        pytest.skip(
            "No baseline snapshot found. Create one with:\n"
            "  python -m AlignAIR.Benchmarking snapshot "
            "--model-dir checkpoints/IGH_S5F_576_Extended "
            "--eval-data tests/data/test/sample_igh_extended.csv "
            "--output-dir tests/data/snapshots/IGH_S5F_576_Extended_baseline"
        )
    from AlignAIR.Benchmarking.snapshot import ModelSnapshot
    return ModelSnapshot.load(str(baseline_path))


@pytest.fixture(scope="session")
def current_snapshot(tmp_path_factory):
    """Create a fresh snapshot from the current code."""
    model_path = _find_model_bundle("IGH_S5F_576_Extended")
    eval_path = _find_eval_data("IGH_S5F_576_Extended")

    if model_path is None:
        pytest.skip("IGH_S5F_576_Extended model bundle not available")
    if eval_path is None:
        pytest.skip("Evaluation data not available")

    from AlignAIR.Benchmarking.snapshot import ModelSnapshot

    output_dir = tmp_path_factory.mktemp("current_snapshot")
    ModelSnapshot.create(
        model_dir=str(model_path),
        eval_data_path=str(eval_path),
        output_dir=str(output_dir),
        batch_size=16,
        include_latent=True,
        include_pipeline=True,
    )
    return ModelSnapshot.load(str(output_dir))


@pytest.fixture(scope="session")
def comparator(baseline_snapshot, current_snapshot):
    """Create a SnapshotComparator with code-change tolerances."""
    from AlignAIR.Benchmarking.compare import SnapshotComparator
    from AlignAIR.Benchmarking.tolerances import CODE_CHANGE_TOLERANCES
    return SnapshotComparator(baseline_snapshot, current_snapshot, CODE_CHANGE_TOLERANCES)


# ---------------------------------------------------------------------------
# Prediction Reproducibility Tests
# ---------------------------------------------------------------------------

class TestPredictionReproducibility:
    """Same model + same data -> identical outputs (detects code-change regressions)."""

    def test_allele_classifications_match(self, comparator):
        """V/D/J allele probability vectors should be identical."""
        results = comparator.compare_predictions()
        allele_results = {k: v for k, v in results.items() if 'allele' in k}
        for key, result in allele_results.items():
            assert result['status'] == 'PASS', (
                f"{key} regression: {result.get('detail', '')}"
            )

    def test_segmentation_positions_match(self, comparator):
        """V/D/J start/end positions should be identical."""
        results = comparator.compare_predictions()
        pos_results = {k: v for k, v in results.items()
                       if k.endswith(('_start', '_end')) and 'logits' not in k}
        for key, result in pos_results.items():
            assert result['status'] == 'PASS', (
                f"{key} regression: {result.get('detail', '')}"
            )

    def test_scalar_outputs_match(self, comparator):
        """Mutation rate, indel count, and productivity should be identical."""
        results = comparator.compare_predictions()
        scalar_keys = ['mutation_rate', 'indel_count', 'productive']
        for key in scalar_keys:
            if key in results:
                assert results[key]['status'] == 'PASS', (
                    f"{key} regression: {results[key].get('detail', '')}"
                )

    def test_no_missing_output_keys(self, baseline_snapshot, current_snapshot):
        """Current snapshot should have all the same output keys as baseline."""
        if 'predictions' not in baseline_snapshot or 'predictions' not in current_snapshot:
            pytest.skip("Predictions not available in both snapshots")
        baseline_keys = set(baseline_snapshot['predictions'].keys())
        current_keys = set(current_snapshot['predictions'].keys())
        missing = baseline_keys - current_keys
        assert not missing, f"Missing prediction keys in current: {missing}"

    def test_output_shapes_match(self, baseline_snapshot, current_snapshot):
        """All prediction output shapes should match."""
        if 'predictions' not in baseline_snapshot or 'predictions' not in current_snapshot:
            pytest.skip("Predictions not available in both snapshots")
        bp = baseline_snapshot['predictions']
        cp = current_snapshot['predictions']
        for key in bp:
            if key in cp:
                assert bp[key].shape == cp[key].shape, (
                    f"{key} shape mismatch: {bp[key].shape} vs {cp[key].shape}"
                )


# ---------------------------------------------------------------------------
# Latent Space Stability Tests
# ---------------------------------------------------------------------------

class TestLatentSpaceStability:
    """Latent representations should be identical for same model weights."""

    def test_v_latent_cosine_similarity(self, comparator):
        """V-gene latent representations should have cosine similarity ~1.0."""
        results = comparator.compare_latent_space()
        if 'v' in results:
            assert results['v']['status'] == 'PASS', (
                f"V latent drift: {results['v'].get('detail', '')}"
            )

    def test_j_latent_cosine_similarity(self, comparator):
        """J-gene latent representations should have cosine similarity ~1.0."""
        results = comparator.compare_latent_space()
        if 'j' in results:
            assert results['j']['status'] == 'PASS', (
                f"J latent drift: {results['j'].get('detail', '')}"
            )

    def test_d_latent_cosine_similarity(self, comparator):
        """D-gene latent representations should have cosine similarity ~1.0."""
        results = comparator.compare_latent_space()
        if 'd' not in results:
            pytest.skip("No D-gene latent in snapshot (light chain model)")
        assert results['d']['status'] == 'PASS', (
            f"D latent drift: {results['d'].get('detail', '')}"
        )

    def test_latent_shapes_preserved(self, baseline_snapshot, current_snapshot):
        """Latent representation dimensions should not change."""
        if 'latent' not in baseline_snapshot or 'latent' not in current_snapshot:
            pytest.skip("Latent not available in both snapshots")
        for gene in baseline_snapshot['latent']:
            if gene in current_snapshot['latent']:
                b_shape = baseline_snapshot['latent'][gene].shape
                c_shape = current_snapshot['latent'][gene].shape
                assert b_shape == c_shape, (
                    f"{gene} latent shape changed: {b_shape} -> {c_shape}"
                )


# ---------------------------------------------------------------------------
# Pipeline Output Reproducibility Tests
# ---------------------------------------------------------------------------

class TestPipelineReproducibility:
    """Full pipeline CSV output should be identical."""

    def test_allele_calls_exact_match(self, comparator):
        """Allele call strings should match exactly."""
        results = comparator.compare_pipeline()
        call_results = {k: v for k, v in results.items() if k.endswith('_call')}
        for key, result in call_results.items():
            assert result['status'] == 'PASS', (
                f"{key}: {result.get('detail', '')}"
            )

    def test_positions_exact_match(self, comparator):
        """Segmentation positions should match exactly."""
        results = comparator.compare_pipeline()
        pos_results = {k: v for k, v in results.items()
                       if any(k.endswith(s) for s in ('_start', '_end'))}
        for key, result in pos_results.items():
            assert result['status'] == 'PASS', (
                f"{key}: {result.get('detail', '')}"
            )

    def test_numeric_outputs_close(self, comparator):
        """Numeric outputs (mutation_rate, indels) should be close."""
        results = comparator.compare_pipeline()
        numeric_results = {k: v for k, v in results.items()
                          if k in ('mutation_rate', 'indels')}
        for key, result in numeric_results.items():
            assert result['status'] == 'PASS', (
                f"{key}: {result.get('detail', '')}"
            )


# ---------------------------------------------------------------------------
# Metric Stability Tests
# ---------------------------------------------------------------------------

class TestMetricStability:
    """Quality metrics should not degrade."""

    def test_prediction_statistics_stable(self, comparator):
        """Prediction statistics should remain stable."""
        results = comparator.compare_metrics()
        for key, result in results.items():
            if result['status'] == 'FAIL':
                pytest.fail(f"Metric regression: {key} - {result.get('detail', '')}")


# ---------------------------------------------------------------------------
# Training Convergence Tests
# ---------------------------------------------------------------------------

class TestTrainingConvergence:
    """Verify that retraining produces equivalent convergence metrics.

    These tests compare training_meta.json between baseline and current snapshots.
    For code-change detection: same training data + same seeds should give identical metrics.
    For model comparison: metrics should be within tolerance of the baseline.
    """

    def test_final_loss_within_tolerance(self, comparator):
        """Final training loss should not regress."""
        results = comparator.compare_training()
        if 'final_loss' in results:
            assert results['final_loss']['status'] == 'PASS', (
                f"Training loss regression: {results['final_loss'].get('detail', '')}"
            )

    def test_best_loss_within_tolerance(self, comparator):
        """Best training loss should not regress."""
        results = comparator.compare_training()
        if 'best_loss' in results:
            assert results['best_loss']['status'] == 'PASS', (
                f"Best loss regression: {results['best_loss'].get('detail', '')}"
            )

    def test_allele_auc_stable(self, comparator):
        """V/D/J allele AUC from training should remain stable."""
        results = comparator.compare_training()
        auc_results = {k: v for k, v in results.items() if 'auc' in k}
        for key, result in auc_results.items():
            assert result['status'] == 'PASS', (
                f"AUC regression: {key} - {result.get('detail', '')}"
            )

    def test_boundary_accuracy_stable(self, comparator):
        """V/D/J boundary accuracy from training should remain stable."""
        results = comparator.compare_training()
        acc_results = {k: v for k, v in results.items() if 'acc' in k}
        for key, result in acc_results.items():
            assert result['status'] == 'PASS', (
                f"Boundary accuracy regression: {key} - {result.get('detail', '')}"
            )


# ---------------------------------------------------------------------------
# Full Report Test
# ---------------------------------------------------------------------------

class TestFullComparison:
    """Run the complete comparison and check overall status."""

    def test_overall_pass(self, comparator):
        """The full comparison should pass with code-change tolerances."""
        result = comparator.compare_all()
        if result.overall_status == 'FAIL':
            from AlignAIR.Benchmarking.report import text_report
            report = text_report(result)
            pytest.fail(f"Regression detected:\n{report}")
