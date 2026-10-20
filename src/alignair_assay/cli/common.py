"""Pieces shared by the commands: argument types, matching summaries and the truth reference of built suites.
Report provenance and report files are ``workflow.scoring``'s, shared with ``all``."""
from __future__ import annotations

import argparse
from pathlib import Path

from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import Reference
from alignair_assay.runners.base import absolute_path
from alignair_assay.truth.io import read_truth_header


def file_name(value: str) -> str:
    """Argument type for a name used as a file-name component (a tool name): no directory part."""
    if not value or Path(value).name != value:
        raise argparse.ArgumentTypeError(f"not a plain file-name component: {value!r}")
    return value


def tool_path(path: str | None) -> str | None:
    """A tool path option (``--igblast-dir``, ``--alignair-checkpoint``) made absolute, ``~`` expanded
    (``runners.base.absolute_path``), so a recorded configuration never depends on where a command was started;
    None stays None."""
    return None if path is None else absolute_path(path)


def comma_list(value: str) -> tuple[str, ...]:
    """The non-empty, stripped items of a comma-separated list, in order and without repeats."""
    return tuple(dict.fromkeys(item.strip() for item in value.split(",") if item.strip()))


def print_matching(tool: str, m: dict) -> None:
    """One line of prediction/truth matching counts for ``tool`` (see ``scoring.pipeline.score_tool``)."""
    print(f"{tool}: {m['scored']} cases scored ({m['excluded']} excluded of {m['truth_cases']}); "
          f"{m['rows']} rows, {m['duplicates']} duplicates, {m['invalid_rows']} invalid rows; "
          f"missing {m['missing']}, extra {m['extra']}, ignored for excluded cases {m['excluded_ignored']}")


def truth_reference(layout: RunLayout, suites: list[str]) -> Reference:
    """The reference the named suites' truth was built against, from its truth headers. ValueError when a suite
    is not built (no ``build.json``), the suites name different references, or the reference's dataconfig no
    longer has the recorded digest."""
    recorded = {}
    for suite in suites:
        if not layout.build_json(suite).is_file():
            raise ValueError(f"suite {suite} is not built in {layout.root} (no {layout.build_json(suite).name}); "
                             f"build it first")
        header = read_truth_header(layout.truth(suite)) or {}
        recorded[suite] = (header.get("reference"), header.get("reference_digest"))
    if len(set(recorded.values())) > 1:
        listed = ", ".join(f"{suite}: {name} ({digest and digest[:12]})" for suite, (name, digest) in recorded.items())
        raise ValueError(f"the suites were built against different references: {listed}")
    name, digest = next(iter(recorded.values()))
    if name is None or digest is None:
        raise ValueError(f"{layout.truth(suites[0])} records no reference digest; rebuild the suite")
    reference = Reference.from_dataconfig(name)
    if reference.digest() != digest:
        raise ValueError(f"the truth was built against {name} with digest {digest[:12]}, but {name} now has digest "
                         f"{reference.digest()[:12]}; rebuild the suites")
    return reference
