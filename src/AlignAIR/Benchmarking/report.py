"""
Human-readable comparison reports from snapshot comparisons.
"""

import json
from pathlib import Path
from typing import Optional

from AlignAIR.Benchmarking.compare import ComparisonResult


def text_report(result: ComparisonResult) -> str:
    """Generate a detailed text report from a ComparisonResult."""
    lines = []
    lines.append("=" * 72)
    lines.append("  AlignAIR Reproducibility Report")
    lines.append("=" * 72)
    lines.append("")

    status_symbol = {
        ComparisonResult.PASS: "[PASS]",
        ComparisonResult.FAIL: "[FAIL]",
        ComparisonResult.WARN: "[WARN]",
    }

    lines.append(f"  Overall Status: {status_symbol.get(result.overall_status, '???')} {result.overall_status}")
    lines.append("")

    for section_name, section_results in result.sections.items():
        lines.append("-" * 72)
        lines.append(f"  {section_name.upper()}")
        lines.append("-" * 72)

        # Count pass/fail/warn
        statuses = [
            v.get('status', '') for v in section_results.values()
            if isinstance(v, dict) and 'status' in v
        ]
        pass_count = statuses.count(ComparisonResult.PASS)
        fail_count = statuses.count(ComparisonResult.FAIL)
        warn_count = statuses.count(ComparisonResult.WARN)
        lines.append(f"  {pass_count} passed, {fail_count} failed, {warn_count} warnings")
        lines.append("")

        for key, val in section_results.items():
            if isinstance(val, dict) and 'status' in val:
                symbol = status_symbol.get(val['status'], '???')
                detail = val.get('detail', '')
                lines.append(f"  {symbol} {key}")
                if detail:
                    lines.append(f"         {detail}")
            else:
                lines.append(f"  {key}: {val}")

        lines.append("")

    lines.append("=" * 72)
    return "\n".join(lines)


def save_report(result: ComparisonResult, output_path: str, format: str = "text") -> str:
    """
    Save a comparison report to file.

    Args:
        result: ComparisonResult from SnapshotComparator.compare_all()
        output_path: File path to save the report.
        format: 'text' or 'json'

    Returns:
        Path to the saved report.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        with open(path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)
    else:
        with open(path, "w") as f:
            f.write(text_report(result))

    return str(path)
