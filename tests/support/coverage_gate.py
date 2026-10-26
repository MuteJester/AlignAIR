"""Enforce measured statement/branch floors for explicitly selected critical files."""
import argparse
import json
from pathlib import Path
import sys


def check_coverage(report, policy):
    if policy.get("schema") != "alignair.coverage-policy.v1" or not policy.get("files"):
        raise ValueError("Invalid or empty coverage policy")
    failures = []
    for name, floors in policy["files"].items():
        if name not in report["files"]:
            failures.append(f"{name}: missing from coverage evidence")
            continue
        summary = report["files"][name]["summary"]
        for kind, covered_key, total_key in (
            ("statements", "covered_lines", "num_statements"),
            ("branches", "covered_branches", "num_branches"),
        ):
            floor = floors[kind]
            if type(floor) not in (int, float) or not 0 <= floor <= 100:
                raise ValueError(f"{name}: invalid {kind} floor")
            covered, total = summary[covered_key], summary[total_key]
            if (type(covered) is not int or type(total) is not int
                    or not 0 <= covered <= total or (kind == "statements" and total == 0)):
                raise ValueError(f"{name}: invalid {kind} evidence")
            percent = 100 * covered / total if total else 100
            if percent < floor:
                failures.append(f"{name}: {kind} {percent:.2f}% < {floor}%")
    return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coverage", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=Path("tests/coverage-policy.json"))
    args = parser.parse_args()
    try:
        failures = check_coverage(json.loads(args.coverage.read_text()),
                                  json.loads(args.policy.read_text()))
    except (OSError, ValueError, KeyError, TypeError) as exc:
        print(f"Invalid coverage evidence: {exc}", file=sys.stderr)
        return 1
    if failures:
        for failure in failures:
            print(failure, file=sys.stderr)
        return 1
    print("Critical-file statement and branch coverage floors passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
