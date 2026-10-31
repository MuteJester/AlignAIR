"""A checkpoint against IgBLAST on the lab's dev suites, paired read by read.

``igblast-dev`` (``run_igblast_dev``) runs IgBLAST once with the assay (``all``, the oracle its second tool) on a copy
of the lab's dev build, so its reads are the screens' reads, and checks afterwards that the run left the builds as
they were. ``against-igblast`` (``run_against_igblast``) diagnoses a checkpoint on the same suites
(``workflow.diagnose.evaluate``: the product pipeline and the truth-substitution chain), with any model options
switched on over its saved weights (an option that adds no weights, such as ``bank_orientation``), and pairs its
per-read scores with IgBLAST's (``pair``): the assay's ``stats.compare`` (an exact McNemar test per 0/1 metric,
cluster-aware where a suite has clusters, Holm across the metrics; Δ = the checkpoint − IgBLAST) and
``stats.read_changes`` (the reads only IgBLAST, and only the checkpoint, gets right), per suite and pooled over every
suite, with the checkpoint's rate under each step of the chain. ``report.md`` leads with the pooled table; the run
directory holds ``report.json`` and the assay's diagnostics. Weights or temperatures tuned on these suites are tuned
on dev: a claim comes from the locked test split.
"""
from __future__ import annotations

import json
import math
import shutil
from dataclasses import replace
from pathlib import Path
from typing import Mapping, Sequence

from alignair_assay.core.layout import RunLayout
from alignair_assay.diagnostics.ablation import VARIANTS
from alignair_assay.metrics import lower_is_better
from alignair_assay.scoring.clusters import cluster_ids
from alignair_assay.scoring.pipeline import load_truth, read_scores, truth_reference
from alignair_assay.stats.compare import compare, read_changes, to_dict
from alignair_assay.truth.io import read_truth_header

from alignair_lab.screen import ARCH_ROOT, DEV_BUILD, assay_layout, built_suites

IGBLAST_DEV = ARCH_ROOT / "igblast_dev"
REPORTS = ARCH_ROOT / "against_igblast"
IGBLAST_DIR = Path("/home/thomas/Desktop/AlignAIR/.private/tools/ncbi-igblast-1.22.0")
TOOL, ARM = "igblast", "shared"
# The paired metrics: the headline, the junction-side boundaries (read and germline), productivity.
METRICS = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "d.top1_in_set", "j.top1_in_set",
           "v.seq_end_exact", "d.seq_start_exact", "d.seq_end_exact", "j.seq_start_exact", "v.germ_end_exact",
           "j.germ_start_exact", "productive.correct")
CHAIN = VARIANTS[1:]           # the truth-substitution steps after the product pipeline itself
_SETTINGS = ("split", "seed", "scale", "reference")
VERDICTS = {"a_better": "IgBLAST better", "b_better": "checkpoint better", "tie_or_inconclusive": "tie"}


# --- the IgBLAST run -------------------------------------------------------------------------------------------------

def dev_settings(dev_build: Path, suites: Sequence[str]) -> dict:
    """The split, seed, scale and reference the dev build's suites were built with (ValueError when they differ)."""
    found = {}
    for suite in suites:
        record = json.loads((Path(dev_build) / "suites" / suite / "build.json").read_text())
        for key in _SETTINGS:
            if key in found and found[key] != record[key]:
                raise ValueError(f"the dev build's suites differ in {key} ({found[key]!r}, {suite}: {record[key]!r})")
            found[key] = record[key]
    return found


def igblast_dev_argv(run_dir: Path, suites: Sequence[str], settings: Mapping, igblast_dir: Path) -> list[str]:
    """The assay's ``all`` arguments that run IgBLAST (and the oracle) on the suites of ``run_dir``."""
    return ["all", "--run-dir", str(run_dir), "--suites", ",".join(suites), "--tools", "igblast,oracle",
            "--split", str(settings["split"]), "--seed", str(settings["seed"]), "--scale", str(settings["scale"]),
            "--reference", str(settings["reference"]), "--igblast-dir", str(igblast_dir)]


def check_same_builds(run_dir: Path, dev_build: Path, suites: Sequence[str]) -> None:
    """ValueError naming each suite whose build in ``run_dir`` is missing or is not the dev build's (other reads)."""
    differing = [suite for suite in suites
                 if not (Path(run_dir) / "suites" / suite / "build.json").is_file()
                 or (Path(run_dir) / "suites" / suite / "build.json").read_bytes()
                 != (Path(dev_build) / "suites" / suite / "build.json").read_bytes()]
    if differing:
        raise ValueError(f"the IgBLAST run {run_dir} has no build, or another one, of {', '.join(differing)} than the "
                         f"dev build {dev_build}: its reads are not the screens' (run igblast-dev)")


def run_igblast_dev(*, run_dir: Path = IGBLAST_DEV, dev_build: Path = DEV_BUILD, suites: Sequence[str] | None = None,
                    igblast_dir: Path = IGBLAST_DIR) -> Path:
    """Copy the dev build's suites into ``run_dir`` (those it does not hold yet), run IgBLAST on them with the assay,
    and check the builds are still the dev build's."""
    from alignair_assay.cli import main as assay
    names = built_suites(Path(dev_build), suites)
    run_dir = Path(run_dir)
    for suite in names:
        if not (run_dir / "suites" / suite).exists():
            shutil.copytree(Path(dev_build) / "suites" / suite, run_dir / "suites" / suite)
    check_same_builds(run_dir, dev_build, names)
    if assay(igblast_dev_argv(run_dir, names, dev_settings(dev_build, names), igblast_dir)) not in (0, None):
        raise RuntimeError(f"the assay's IgBLAST run in {run_dir} failed (see its report)")
    check_same_builds(run_dir, dev_build, names)
    return run_dir


# --- pairing ---------------------------------------------------------------------------------------------------------

def _compared(theirs: dict, ours: dict, clusters: Mapping[str, str], metrics: Sequence[str]) -> dict:
    low = {metric for metric in metrics if lower_is_better(metric)}
    return {"comparison": to_dict(compare(theirs, ours, list(metrics), lower_is_better=low, clusters=clusters)),
            "changes": read_changes(theirs, ours, list(metrics), lower_is_better=low)}


def _chain(cases: Mapping[str, dict], metrics: Sequence[str]) -> dict:
    """The checkpoint's mean of each metric under each truth-substitution step (None: no read scores it)."""
    chain = {}
    for variant in CHAIN:
        rates = {}
        for metric in metrics:
            values = [entry["variants"].get(variant, {}).get(metric) for entry in cases.values()]
            values = [value for value in values if value is not None and math.isfinite(value)]
            rates[metric] = math.fsum(values) / len(values) if values else None
        chain[variant] = rates
    return chain


def pair(igblast: Mapping[str, dict], candidate: Mapping[str, dict], clusters: Mapping[str, Mapping[str, str]], *,
         metrics: Sequence[str] = METRICS) -> dict:
    """IgBLAST's per-read scores (suite -> case -> scores) against the checkpoint's (suite -> case -> its
    diagnostics' ``head``, ``baseline`` and ``variants``), per suite and pooled: ``comparison`` (``stats.compare``,
    a = IgBLAST, b = the checkpoint), ``changes`` (``stats.read_changes``: ``broken`` = only IgBLAST right, ``fixed`` =
    only the checkpoint right) and, per suite, the checkpoint's ``chain``. ValueError when IgBLAST has no scores of a
    read the checkpoint was diagnosed on: a comparison never pairs a subset."""
    suites, pooled_theirs, pooled_ours, pooled_clusters = {}, {}, {}, {}
    for suite, cases in candidate.items():
        ours = {case: entry["baseline"] for case, entry in cases.items() if entry["baseline"] is not None}
        missing = sorted(set(ours) - set(igblast.get(suite, {})))
        if missing:
            raise ValueError(f"suite {suite}: IgBLAST has no scores of {len(missing)} reads the checkpoint has (e.g. "
                             f"{', '.join(missing[:3])}): a comparison never pairs a subset")
        theirs = {case: igblast[suite][case] for case in ours}
        suite_clusters = dict(clusters.get(suite, {}))
        suites[suite] = {"reads": len(ours), **_compared(theirs, ours, suite_clusters, metrics),
                         "chain": _chain(cases, metrics)}
        pooled_theirs.update(theirs)
        pooled_ours.update(ours)
        pooled_clusters.update(suite_clusters)
    return {"metrics": list(metrics), "suites": suites,
            "pooled": {"reads": len(pooled_ours), **_compared(pooled_theirs, pooled_ours, pooled_clusters, metrics)}}


# --- the report ------------------------------------------------------------------------------------------------------

def _rate(value) -> str:
    return "–" if value is None else f"{100 * value:.1f}"


def _comparison_table(found: dict, metrics: Sequence[str]) -> list[str]:
    lines = ["| Metric | IgBLAST | checkpoint | Δ | 95% CI | p | verdict | only IgBLAST / only checkpoint |",
             "|---|---|---|---|---|---|---|---|"]
    for metric in metrics:
        entry = found["comparison"].get(metric) or {}
        if not entry.get("n_paired"):
            continue
        changes = found["changes"]["metrics"].get(metric, {})
        delta, ci = entry["delta"], entry.get("ci95")
        lines.append(f"| {metric} | {_rate(entry['mean_a_paired'])} | {_rate(entry['mean_b_paired'])} | "
                     f"{100 * delta:+.1f} pp | " + (f"[{100 * ci[0]:+.1f}, {100 * ci[1]:+.1f}]" if ci else "–")
                     + f" | {entry['p_value']:.2g} | {VERDICTS[entry['verdict']]} | "
                     f"{changes.get('broken', '–')} / {changes.get('fixed', '–')} |")
    return lines


def report_markdown(result: dict, *, name: str) -> str:
    metrics = result["metrics"]
    lines = [f"# {name} against IgBLAST", "",
             "Paired read by read on the lab's dev suites (the same reads). Δ = the checkpoint − IgBLAST; p: exact "
             "McNemar (cluster-aware where a suite has clusters), verdicts Holm-corrected across the metrics; the last "
             "column counts the reads only IgBLAST, and only the checkpoint, gets right. Dev split: a claim comes "
             "from the locked test split.", "",
             f"## All suites ({result['pooled']['reads']} reads)", "", *_comparison_table(result["pooled"], metrics)]
    for suite, found in result["suites"].items():
        lines += ["", f"## {suite} ({found['reads']} reads)", "", *_comparison_table(found, metrics), "",
                  "The checkpoint when each stage is given its truth (cumulative):", "",
                  "| Metric | " + " | ".join(CHAIN) + " |", "|---|" + "---|" * len(CHAIN)]
        lines += [f"| {metric} | " + " | ".join(_rate(found["chain"][variant][metric]) for variant in CHAIN) + " |"
                  for metric in metrics]
    return "\n".join(lines) + "\n"


# --- the run ---------------------------------------------------------------------------------------------------------

def _model(checkpoint: Path, options: Mapping, device: str):
    """``(model, reference_set)``: the checkpoint's weights in a model built with ``options`` over its config."""
    import alignair_lab.architectures  # noqa: F401  (a lab checkpoint's architecture)
    from alignair.core import AlignAIR
    from alignair.model_file import load_model
    loaded = load_model(checkpoint, device=device)
    model = AlignAIR(replace(loaded.config, **options)).to(device)
    model.load_state_dict(loaded.model.state_dict())
    model.bind_reference(loaded.reference)
    return model.eval(), loaded.reference


def run_against_igblast(name: str, checkpoint: Path, *, options: Mapping | None = None,
                        igblast_run: Path = IGBLAST_DEV, dev_build: Path = DEV_BUILD,
                        suites: Sequence[str] | None = None, device: str = "cuda", out_root: Path = REPORTS,
                        metrics: Sequence[str] = METRICS) -> str:
    """Diagnose ``checkpoint`` (with ``options``) on the dev suites into ``out_root/name`` and pair it with IgBLAST's
    run on the same reads; write ``report.json`` and ``report.md`` there and return the report."""
    from alignair_assay.workflow.diagnose import evaluate, read_case_scores
    names = built_suites(Path(dev_build), suites)
    check_same_builds(igblast_run, dev_build, names)
    scores_of = RunLayout(Path(igblast_run))
    for suite in names:
        if not scores_of.tool_scores(suite, ARM, TOOL).is_file():
            raise FileNotFoundError(f"the IgBLAST run {igblast_run} has no scores of {suite} (run igblast-dev)")
    out = Path(out_root) / name
    out.mkdir(parents=True)                                   # a name in use is refused
    layout = assay_layout(out / "assay", Path(dev_build), names)
    model, reference_set = _model(Path(checkpoint), dict(options or {}), device)
    result = evaluate((model, reference_set), layout, names, device=device, variants=VARIANTS, profile=False,
                      label=name)
    reference = truth_reference(layout, names)
    candidate, clusters, theirs = {}, {}, {}
    for suite in names:
        candidate[suite] = read_case_scores(layout, result.model_key, suite)
        header = read_truth_header(layout.truth(suite)) or {}
        clusters[suite] = cluster_ids(load_truth(layout.truth(suite), header.get("reference"), reference)[1])
        theirs[suite] = read_scores(scores_of.tool_scores(suite, ARM, TOOL))
    paired = pair(theirs, candidate, clusters, metrics=metrics)
    record = {"name": name, "checkpoint": str(checkpoint), "options": dict(options or {}),
              "model_key": result.model_key, "igblast_run": str(igblast_run), "dev_build": str(dev_build), **paired}
    (out / "report.json").write_text(json.dumps(record, indent=1, default=str))
    text = report_markdown(paired, name=name)
    (out / "report.md").write_text(text)
    return text
