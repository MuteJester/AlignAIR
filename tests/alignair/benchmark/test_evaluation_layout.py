from pathlib import Path

import alignair.benchmark.evaluation as evaluation
from alignair.benchmark.evaluation.adapters import case_to_prediction
from alignair.inference.calibration import _set_stats
from alignair.benchmark.evaluation.performance import PERFORMANCE_GLOBAL_KEYS
from alignair.benchmark.evaluation.report_validation import validate_benchmark_report_contract


def test_evaluation_root_has_no_dangling_implementation_modules() -> None:
    root = Path(evaluation.__file__).parent

    assert {path.name for path in root.glob("*.py")} == {"__init__.py"}
    assert {
        "adapters",
        "audit",
        "context",
        "contract",
        "diagnostics",
        "matching",
        "metrics",
        "model_adapters",
        "online",
        "performance",
        "report",
        "report_validation",
        "runner",
        "uncertainty",
    } <= {path.name for path in root.iterdir() if path.is_dir()}


def test_legacy_evaluation_module_imports_still_resolve() -> None:
    assert callable(case_to_prediction)
    assert callable(_set_stats)
    assert PERFORMANCE_GLOBAL_KEYS
    assert callable(validate_benchmark_report_contract)
