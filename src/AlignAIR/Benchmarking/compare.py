"""
Compare two reproducibility snapshots and produce a structured report.

Supports three comparison modes:
- code-change: Same model weights, code refactored. Very strict tolerances.
- default: General-purpose comparison.
- model-comparison: Two different models. Loose tolerances.
"""

import logging
from typing import Optional

import numpy as np
import pandas as pd

from AlignAIR.Benchmarking.tolerances import DEFAULT_TOLERANCES, get_tolerances

logger = logging.getLogger(__name__)


class ComparisonResult:
    """Structured result from a snapshot comparison."""

    PASS = "PASS"
    FAIL = "FAIL"
    WARN = "WARN"

    def __init__(self):
        self.sections = {}
        self.overall_status = self.PASS

    def add_section(self, name: str, results: dict):
        self.sections[name] = results
        # Propagate worst status
        for item in results.values():
            if isinstance(item, dict) and 'status' in item:
                if item['status'] == self.FAIL:
                    self.overall_status = self.FAIL
                elif item['status'] == self.WARN and self.overall_status != self.FAIL:
                    self.overall_status = self.WARN

    def to_dict(self) -> dict:
        return {
            "status": self.overall_status,
            **self.sections,
        }

    def summary(self) -> str:
        """Human-readable summary."""
        lines = [f"Overall: {self.overall_status}"]
        for section_name, results in self.sections.items():
            lines.append(f"\n  [{section_name}]")
            for key, val in results.items():
                if isinstance(val, dict) and 'status' in val:
                    status = val['status']
                    detail = val.get('detail', '')
                    lines.append(f"    {key}: {status}  {detail}")
                else:
                    lines.append(f"    {key}: {val}")
        return "\n".join(lines)


class SnapshotComparator:
    """Compare two model snapshots and produce a structured pass/fail report."""

    def __init__(
        self,
        baseline: dict,
        current: dict,
        tolerances: Optional[dict] = None,
    ):
        """
        Args:
            baseline: Dict from ModelSnapshot.load() for the reference snapshot.
            current: Dict from ModelSnapshot.load() for the snapshot to compare.
            tolerances: Tolerance dict. Defaults to DEFAULT_TOLERANCES.
        """
        self.baseline = baseline
        self.current = current
        self.tol = tolerances or DEFAULT_TOLERANCES

    @classmethod
    def from_dirs(cls, baseline_dir: str, current_dir: str,
                  tolerance_profile: str = "default") -> "SnapshotComparator":
        """Convenience constructor from snapshot directory paths."""
        from AlignAIR.Benchmarking.snapshot import ModelSnapshot
        baseline = ModelSnapshot.load(baseline_dir)
        current = ModelSnapshot.load(current_dir)
        tolerances = get_tolerances(tolerance_profile)
        return cls(baseline, current, tolerances)

    def compare_all(self) -> ComparisonResult:
        """Run all comparison checks and return a structured result."""
        result = ComparisonResult()

        if 'predictions' in self.baseline and 'predictions' in self.current:
            result.add_section("predictions", self.compare_predictions())

        if 'latent' in self.baseline and 'latent' in self.current:
            result.add_section("latent_space", self.compare_latent_space())

        if 'metrics' in self.baseline and 'metrics' in self.current:
            result.add_section("metrics", self.compare_metrics())

        if 'pipeline_output' in self.baseline and 'pipeline_output' in self.current:
            result.add_section("pipeline", self.compare_pipeline())

        if 'training_meta' in self.baseline and 'training_meta' in self.current:
            result.add_section("training", self.compare_training())

        return result

    def compare_training(self) -> dict:
        """
        Compare training metadata between baseline and current.

        Checks convergence indicators from training_meta.json:
        - Final loss / best loss
        - Per-component losses
        - AUC scores
        - Boundary accuracy metrics
        """
        bt = self.baseline['training_meta']
        ct = self.current['training_meta']
        tol = self.tol.get('training', self.tol.get('metrics', {}))
        results = {}

        # Compare top-level training scalars
        for key in ['best_loss', 'final_loss']:
            if key in bt and key in ct and bt[key] is not None and ct[key] is not None:
                b_val = float(bt[key])
                c_val = float(ct[key])
                # Use relative tolerance for loss values
                rtol = tol.get('loss_rtol', 0.10)
                rel_diff = abs(b_val - c_val) / max(abs(b_val), 1e-10)
                status = ComparisonResult.PASS if rel_diff <= rtol else ComparisonResult.FAIL
                results[key] = {
                    "status": status,
                    "baseline": b_val,
                    "current": c_val,
                    "rel_diff": rel_diff,
                    "tolerance": rtol,
                    "detail": f"baseline={b_val:.4f} current={c_val:.4f} rel_diff={rel_diff:.4f} (rtol={rtol})",
                }

        # Compare metrics_summary (the detailed per-component metrics)
        b_metrics = bt.get('metrics_summary', {})
        c_metrics = ct.get('metrics_summary', {})

        for key in b_metrics:
            if key not in c_metrics:
                continue
            b_val = b_metrics[key]
            c_val = c_metrics[key]
            if not isinstance(b_val, (int, float)) or not isinstance(c_val, (int, float)):
                continue

            diff = abs(b_val - c_val)

            # Determine appropriate tolerance based on metric type
            if 'auc' in key:
                threshold = tol.get('accuracy_atol', 0.02)
            elif 'acc' in key:
                threshold = tol.get('accuracy_atol', 0.02)
            elif 'mae' in key:
                threshold = tol.get('boundary_mae_atol', 1.0)
            elif 'loss' in key:
                # Use relative tolerance for loss components
                rtol = tol.get('loss_rtol', 0.10)
                rel_diff = abs(b_val - c_val) / max(abs(b_val), 1e-10)
                status = ComparisonResult.PASS if rel_diff <= rtol else ComparisonResult.FAIL
                results[f"metrics_summary/{key}"] = {
                    "status": status,
                    "baseline": b_val,
                    "current": c_val,
                    "rel_diff": rel_diff,
                    "tolerance": rtol,
                    "detail": f"baseline={b_val:.4f} current={c_val:.4f} rel_diff={rel_diff:.4f} (rtol={rtol})",
                }
                continue
            else:
                threshold = tol.get('accuracy_atol', 0.02)

            status = ComparisonResult.PASS if diff <= threshold else ComparisonResult.FAIL
            results[f"metrics_summary/{key}"] = {
                "status": status,
                "baseline": b_val,
                "current": c_val,
                "diff": diff,
                "tolerance": threshold,
                "detail": f"baseline={b_val:.4f} current={c_val:.4f} diff={diff:.4f} (tol={threshold})",
            }

        return results

    def compare_predictions(self) -> dict:
        """
        Compare raw model prediction outputs between baseline and current.

        Checks:
        - Allele classification arrays: max/mean absolute difference
        - Position outputs: max/mean absolute difference
        - Scalar outputs: max/mean absolute difference
        """
        bp = self.baseline['predictions']
        cp = self.current['predictions']
        tol = self.tol.get('predictions', {})

        results = {}

        # Categorize output keys
        allele_keys = [k for k in bp if 'allele' in k]
        position_keys = [k for k in bp if k.endswith(('_start', '_end')) or k.endswith('_logits')]
        scalar_keys = [k for k in bp if k in ('mutation_rate', 'indel_count', 'productive')]

        for key in allele_keys:
            if key not in cp:
                results[key] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue
            results[key] = _compare_arrays(
                bp[key], cp[key],
                atol=tol.get('allele_classification_atol', 1e-4),
                label=key,
            )

        for key in position_keys:
            if key not in cp:
                results[key] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue
            results[key] = _compare_arrays(
                bp[key], cp[key],
                atol=tol.get('position_atol', 1e-4),
                label=key,
            )

        for key in scalar_keys:
            if key not in cp:
                results[key] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue
            results[key] = _compare_arrays(
                bp[key], cp[key],
                atol=tol.get('scalar_atol', 1e-3),
                label=key,
            )

        return results

    def compare_latent_space(self) -> dict:
        """
        Compare latent representations between baseline and current.

        Checks per gene:
        - Per-sample cosine similarity (mean, min)
        - Mean vector L2 distance
        """
        bl = self.baseline['latent']
        cl = self.current['latent']
        tol = self.tol.get('latent', {})
        results = {}

        for gene in bl:
            if gene not in cl:
                results[gene] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue

            b = bl[gene]
            c = cl[gene]

            if b.shape != c.shape:
                results[gene] = {
                    "status": ComparisonResult.FAIL,
                    "detail": f"shape mismatch: {b.shape} vs {c.shape}",
                }
                continue

            # Per-sample cosine similarity
            cos_sim = _batch_cosine_similarity(b, c)
            mean_cos = float(np.mean(cos_sim))
            min_cos = float(np.min(cos_sim))

            # Mean vector L2 distance
            mean_b = np.mean(b, axis=0)
            mean_c = np.mean(c, axis=0)
            mean_l2 = float(np.linalg.norm(mean_b - mean_c))

            cos_threshold = tol.get('cosine_similarity_min', 0.999)
            l2_threshold = tol.get('mean_vector_l2_max', 1e-3)

            status = ComparisonResult.PASS
            if min_cos < cos_threshold or mean_l2 > l2_threshold:
                status = ComparisonResult.FAIL

            results[gene] = {
                "status": status,
                "mean_cosine_similarity": mean_cos,
                "min_cosine_similarity": min_cos,
                "mean_vector_l2_distance": mean_l2,
                "detail": f"cos_sim: mean={mean_cos:.6f} min={min_cos:.6f}, mean_l2={mean_l2:.6f}",
            }

        return results

    def compare_metrics(self) -> dict:
        """
        Compare quality metrics between baseline and current.

        Checks each metric against its configured tolerance.
        """
        bm = self.baseline['metrics']
        cm = self.current['metrics']
        tol = self.tol.get('metrics', {})
        results = {}

        for key in bm:
            if key not in cm:
                results[key] = {"status": ComparisonResult.WARN, "detail": "missing in current"}
                continue

            b_vals = bm[key]
            c_vals = cm[key]

            if not isinstance(b_vals, dict) or not isinstance(c_vals, dict):
                continue

            # Compare each sub-metric
            for sub_key in b_vals:
                if sub_key not in c_vals:
                    continue

                b_val = b_vals[sub_key]
                c_val = c_vals[sub_key]

                if not isinstance(b_val, (int, float)) or not isinstance(c_val, (int, float)):
                    continue

                diff = abs(b_val - c_val)
                metric_key = f"{key}/{sub_key}"

                # Determine tolerance
                if 'accuracy' in sub_key or 'match' in sub_key:
                    threshold = tol.get('accuracy_atol', 0.02)
                elif 'mae' in sub_key:
                    threshold = tol.get('boundary_mae_atol', 1.0)
                else:
                    threshold = tol.get('accuracy_atol', 0.02)

                status = ComparisonResult.PASS if diff <= threshold else ComparisonResult.FAIL
                results[metric_key] = {
                    "status": status,
                    "baseline": b_val,
                    "current": c_val,
                    "diff": diff,
                    "tolerance": threshold,
                    "detail": f"baseline={b_val:.4f} current={c_val:.4f} diff={diff:.4f} (tol={threshold})",
                }

        return results

    def compare_pipeline(self) -> dict:
        """
        Compare full pipeline CSV outputs.

        Checks:
        - Allele call exact match rate
        - Position exact match rate
        - Numeric column closeness
        """
        bp = self.baseline['pipeline_output']
        cp = self.current['pipeline_output']
        tol = self.tol.get('pipeline', {})
        results = {}

        if len(bp) != len(cp):
            return {"row_count": {
                "status": ComparisonResult.FAIL,
                "detail": f"row count mismatch: {len(bp)} vs {len(cp)}",
            }}

        # Allele call columns
        call_cols = [c for c in bp.columns if c.endswith('_call')]
        for col in call_cols:
            if col not in cp.columns:
                results[col] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue
            match_rate = float((bp[col] == cp[col]).mean())
            threshold = tol.get('allele_call_match_rate_min', 0.98)
            status = ComparisonResult.PASS if match_rate >= threshold else ComparisonResult.FAIL
            results[col] = {
                "status": status,
                "match_rate": match_rate,
                "threshold": threshold,
                "detail": f"match_rate={match_rate:.4f} (threshold={threshold})",
            }

        # Position columns
        pos_cols = [c for c in bp.columns if any(
            c.endswith(s) for s in ('_start', '_end', '_sequence_start', '_sequence_end',
                                     '_germline_start', '_germline_end')
        )]
        for col in pos_cols:
            if col not in cp.columns:
                results[col] = {"status": ComparisonResult.FAIL, "detail": "missing in current"}
                continue
            try:
                b_vals = bp[col].astype(float)
                c_vals = cp[col].astype(float)
                exact_match = float((b_vals == c_vals).mean())
                threshold = tol.get('position_exact_match_min', 0.95)
                status = ComparisonResult.PASS if exact_match >= threshold else ComparisonResult.FAIL
                results[col] = {
                    "status": status,
                    "exact_match_rate": exact_match,
                    "threshold": threshold,
                    "detail": f"exact_match={exact_match:.4f} (threshold={threshold})",
                }
            except (ValueError, TypeError):
                results[col] = {"status": ComparisonResult.WARN, "detail": "non-numeric values"}

        # Numeric columns
        numeric_cols = ['mutation_rate', 'indels']
        for col in numeric_cols:
            if col not in bp.columns or col not in cp.columns:
                continue
            try:
                atol_val = tol.get('numeric_atol', 1e-3)
                close = np.allclose(bp[col].values, cp[col].values, atol=atol_val, equal_nan=True)
                max_diff = float(np.nanmax(np.abs(bp[col].values - cp[col].values)))
                status = ComparisonResult.PASS if close else ComparisonResult.FAIL
                results[col] = {
                    "status": status,
                    "max_diff": max_diff,
                    "atol": atol_val,
                    "detail": f"max_diff={max_diff:.6f} (atol={atol_val})",
                }
            except (ValueError, TypeError):
                results[col] = {"status": ComparisonResult.WARN, "detail": "comparison error"}

        return results


# ---------------------------------------------------------------------------
# Utility functions
# ---------------------------------------------------------------------------

def _compare_arrays(baseline: np.ndarray, current: np.ndarray, atol: float, label: str) -> dict:
    """Compare two numpy arrays and return a status dict."""
    if baseline.shape != current.shape:
        return {
            "status": ComparisonResult.FAIL,
            "detail": f"{label}: shape mismatch {baseline.shape} vs {current.shape}",
        }

    diff = np.abs(baseline - current)
    max_diff = float(np.max(diff))
    mean_diff = float(np.mean(diff))
    is_close = max_diff <= atol

    status = ComparisonResult.PASS if is_close else ComparisonResult.FAIL
    return {
        "status": status,
        "max_diff": max_diff,
        "mean_diff": mean_diff,
        "atol": atol,
        "detail": f"max_diff={max_diff:.2e} mean_diff={mean_diff:.2e} (atol={atol:.0e})",
    }


def _batch_cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Compute per-row cosine similarity between two matrices."""
    # Normalize rows
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)

    # Avoid division by zero
    a_norm = np.where(a_norm == 0, 1, a_norm)
    b_norm = np.where(b_norm == 0, 1, b_norm)

    a_normalized = a / a_norm
    b_normalized = b / b_norm

    # Dot product per row
    return np.sum(a_normalized * b_normalized, axis=1)
