"""``show``: why did this read break? One read looked up in a run directory, every tool side by side.

A pure lookup over the files a run already wrote: it reads, never writes, and never runs a tool or the model.

- ``show_case(layout, case_id)``, a simulated case of a run directory ``all`` or ``build`` made: the ``case`` (its
  suite, stratum, presented sequence and the simulator's calls when its record holds them), its observable
  ``truth`` (``truth.io``'s stored form), each tool's AIRR row (``AIRR_FIELDS``, as the tool wrote them) and
  per-metric scores on the arm, the metrics on which the tools' scores ``differ``, and each diagnosed model's stored
  per-case scores of the case (``diagnose``: its head metrics and end-to-end baseline, under its model key, with
  whether its record still names the per-case file; no replay).
- ``show_read(layout, donor, read_id)``, a read of a ``real`` run directory: the ``read`` (sequence, isotype,
  protocol, masked primers), its ``RealTruth`` (the known and reference-exact answers, the clonotype, the anchor),
  each tool's AIRR row and per-read classes on the arm (``real/<donor>/scores/<arm>/<tool>.scores.jsonl.gz``: its R1
  class ``v``, ``v_support``, ...), and where the tools ``differ``.

Tools are those whose predictions of the suite (or donor) and arm exist, the oracle left out unless named; a named
tool without predictions there is refused naming those that ran. An unknown case, read or donor is refused
(ValueError) naming the ``NEAREST`` nearest ids (``nearest``).
"""
from __future__ import annotations

import csv
import difflib
import math
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path

from alignair_assay.core.cases import read_cases, read_fasta
from alignair_assay.core.jsonl import read_jsonl
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import sha256_file
from alignair_assay.core.records import read_json
from alignair_assay.scoring.pipeline import read_scores

# The AIRR fields shown side by side, as each tool wrote them (1-based, inclusive coordinates).
AIRR_FIELDS = ("rev_comp", "v_call", "d_call", "j_call", "v_sequence_start", "v_sequence_end", "d_sequence_start",
               "d_sequence_end", "j_sequence_start", "j_sequence_end", "junction", "junction_aa", "productive")
# The simulator's calls shown with a simulated case, when its record holds them.
SIMULATED = ("v_call", "d_call", "j_call")
# How many nearest ids an unknown one is refused with.
NEAREST = 3
ORACLE = "oracle"


def nearest(name: str, ids: Iterable[str], n: int = NEAREST) -> list[str]:
    """The ``n`` ids of ``ids`` most like ``name`` (``difflib``'s similarity ratio), most alike first."""
    return difflib.get_close_matches(name, list(ids), n=n, cutoff=0.0)


def show_case(layout: RunLayout, case_id: str, *, tools: Sequence[str] | None = None, arm: str = "shared") -> dict:
    """The simulated case ``case_id`` of ``layout`` (see the module docstring): ``kind`` ``sim``, ``run_dir``,
    ``suite``, ``arm``, ``case``, ``truth``, ``tools`` (tool -> its AIRR ``row``, None without a row for the case, and
    ``scores``, None when unscored), ``differ`` (the metrics whose scores differ between the tools) and ``diagnose``
    (model key -> ``current``, ``head``, ``baseline``, and ``differ`` across the keys under ``diagnose_differ``).
    ValueError for an unknown case (naming the nearest) or a named tool without predictions."""
    suites = built_suites(layout)
    suite = case_id.split(":", 1)[0]
    cases = {case["case_id"]: case for case in read_cases(layout.cases(suite))} if suite in suites else {}
    if case_id not in cases:
        known = cases or {case["case_id"]: None for name in suites for case in read_cases(layout.cases(name))}
        where = layout.cases(suite) if suite in suites else f"the suites built in {layout.root}"
        raise ValueError(f"no case {case_id!r} in {where}; the nearest: {', '.join(nearest(case_id, known)) or 'none'}")
    case = cases[case_id]
    truth = next((row for row in read_jsonl(layout.truth(suite)) if row.get("case_id") == case_id), None)
    ran = sorted(path.name for path in (layout.root / "runs").glob("*")
                 if layout.predictions(path.name, suite, arm).is_file()) if (layout.root / "runs").is_dir() else []
    chosen = _chosen(tools, ran, f"{suite}/{arm}")
    found = {}
    for tool in chosen:
        scores = layout.tool_scores(suite, arm, tool)
        found[tool] = {"row": airr_row(layout.predictions(tool, suite, arm), case_id),
                       "scores": read_scores(scores).get(case_id) if scores.is_file() else None}
    record = case.get("record") or {}
    diagnosed = diagnosed_scores(layout, suite, case_id)
    return {"kind": "sim", "run_dir": str(layout.root), "suite": suite, "arm": arm,
            "case": {"case_id": case_id, "suite": suite, "stratum": case.get("stratum"),
                     "orientation": case.get("orientation"), "sequence": case.get("sequence"),
                     "labels": case.get("labels") or {},
                     "simulated": {name: record[name] for name in SIMULATED if name in record} or None},
            "truth": truth, "tools": found,
            "differ": differing({tool: entry["scores"] for tool, entry in found.items()}),
            "diagnose": diagnosed,
            "diagnose_differ": differing({key: entry["head"] | (entry["baseline"] or {})
                                          for key, entry in diagnosed.items()})}


def show_read(layout: RunLayout, donor: str, read_id: str, *, tools: Sequence[str] | None = None,
              arm: str = "shared") -> dict:
    """The read ``read_id`` of ``donor`` in the real run directory ``layout`` (see the module docstring): ``kind``
    ``real``, ``run_dir``, ``donor``, ``arm``, ``read``, ``truth`` (``real.truth.truth_to_json``'s form), ``tools``
    (tool -> its AIRR ``row`` and per-read ``scores``, each None when missing) and ``differ``. ValueError for an
    unknown donor (naming the donors), read (naming the nearest) or a named tool without predictions."""
    donors = sorted(path.name for path in layout.real_root.glob("*") if layout.real_truth(path.name).is_file()) \
        if layout.real_root.is_dir() else []
    if donor not in donors:
        raise ValueError(f"no donor {donor!r} in {layout.real_root} with a truth; donors: "
                         f"{', '.join(donors) or 'none'}")
    truths = {row["read_id"]: row for row in read_jsonl(layout.real_truth(donor))}
    if read_id not in truths:
        raise ValueError(f"no read {read_id!r} of donor {donor} in {layout.real_truth(donor)}; the nearest: "
                         f"{', '.join(nearest(read_id, truths)) or 'none'}")
    sequence = dict(read_fasta(layout.real_reads(donor))).get(read_id) if layout.real_reads(donor).is_file() else None
    meta = _tsv_row(layout.real_reads_meta(donor), read_id) if layout.real_reads_meta(donor).is_file() else None
    runs = layout.real_donor_dir(donor) / "runs"
    ran = sorted(path.name for path in runs.glob("*")
                 if layout.real_predictions(path.name, donor, arm).is_file()) if runs.is_dir() else []
    chosen = _chosen(tools, ran, f"{donor}/{arm}")
    found = {}
    for tool in chosen:
        scores = layout.real_scores(donor, arm, tool)
        row = next((row for row in read_jsonl(scores) if row.get("read_id") == read_id), None) \
            if scores.is_file() else None
        found[tool] = {"row": airr_row(layout.real_predictions(tool, donor, arm), read_id),
                       "scores": None if row is None else {key: value for key, value in row.items()
                                                           if key != "read_id"}}
    return {"kind": "real", "run_dir": str(layout.root), "donor": donor, "arm": arm,
            "read": {"read_id": read_id, "sequence": sequence, **{key: (meta or {}).get(key) for key in (
                "isotype", "protocol", "primer_masked", "consensus_count", "dupcount")}},
            "truth": truths[read_id], "tools": found,
            "differ": differing({tool: entry["scores"] for tool, entry in found.items()})}


def built_suites(layout: RunLayout) -> list[str]:
    """The suites of ``layout`` with a cases file, sorted."""
    root = layout.root / "suites"
    return sorted(path.name for path in root.glob("*") if layout.cases(path.name).is_file()) if root.is_dir() else []


def airr_row(path: Path, sequence_id: str) -> dict | None:
    """The ``AIRR_FIELDS`` of the row of ``sequence_id`` in the AIRR TSV ``path`` (a field the file lacks is
    absent); None when no row has that id."""
    row = _tsv_row(path, sequence_id)
    return None if row is None else {name: row[name] for name in AIRR_FIELDS if name in row}


def diagnosed_scores(layout: RunLayout, suite: str, case_id: str) -> dict[str, dict]:
    """Per model key diagnosed on ``suite`` in ``layout`` (sorted), the case's stored per-case scores: ``head``,
    ``baseline`` (None when no variant ran) and ``current`` (False when the suite record no longer names the per-case
    file by its SHA-256, so the scores may be another configuration's). A model whose per-case file lacks the case
    is left out."""
    found = {}
    root = layout.diagnostics_root
    for key in sorted(path.name for path in root.glob("*")) if root.is_dir() else []:
        cases = layout.diagnostics_suite_cases(key, suite)
        if not cases.is_file():
            continue
        line = next((line for line in read_jsonl(cases) if line.get("case_id") == case_id), None)
        if line is None:
            continue
        record = read_json(layout.diagnostics_suite_json(key, suite)) or {}
        stored = (record.get("case_scores") or {}).get("sha256")
        found[key] = {"current": stored == sha256_file(cases), "head": _decoded(line.get("head")) or {},
                      "baseline": _decoded(line.get("baseline"))}
    return found


def differing(scores: Mapping[str, Mapping | None]) -> list[str]:
    """The metrics whose values are not the same for every side of ``scores`` (side -> metric -> value; NaN equals
    NaN, an absent metric differs from a present one), sorted; none with fewer than two sides."""
    sides = [side or {} for side in scores.values()]
    if len(sides) < 2:
        return []
    metrics = sorted({metric for side in sides for metric in side})
    return [metric for metric in metrics if len({_key(side.get(metric, _ABSENT)) for side in sides}) > 1]


_ABSENT = object()


def _key(value):
    """A hashable stand-in for a score: NaN is one value, absence another."""
    if value is _ABSENT:
        return ("absent",)
    if isinstance(value, float) and math.isnan(value):
        return ("nan",)
    return ("value", repr(value))


def _chosen(tools: Sequence[str] | None, ran: list[str], where: str) -> list[str]:
    """``tools`` (each must have run), or every tool that ran but the oracle."""
    if tools is None:
        return [tool for tool in ran if tool != ORACLE]
    missing = [tool for tool in tools if tool not in ran]
    if missing:
        raise ValueError(f"no predictions of tool {missing[0]!r} on {where}; tools that ran: "
                         f"{', '.join(ran) or 'none'}")
    return list(dict.fromkeys(tools))


def _tsv_row(path: Path, sequence_id: str) -> dict | None:
    """The row of a TSV whose ``sequence_id`` is ``sequence_id``; None without one."""
    with open(path, newline="", encoding="utf-8") as fh:
        return next((row for row in csv.DictReader(fh, delimiter="\t") if row.get("sequence_id") == sequence_id),
                    None)


def _decoded(scores: Mapping | None) -> dict | None:
    """Stored per-case scores with null read back as NaN."""
    if scores is None:
        return None
    return {name: math.nan if value is None else value for name, value in scores.items()}
