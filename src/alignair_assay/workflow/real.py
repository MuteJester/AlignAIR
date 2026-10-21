"""``real``: the real-data pilot from raw reads to scored, cluster-aware comparisons, resumably (P3a design §6-7, 10).

``run_real(cfg, layout)`` runs each donor of ``cfg`` through the steps of ``workflow.real_steps`` (fetch, genotype and
capture, preprocess, truth, the tools' runs) and scores it with ``workflow.real_score``, then pools the donors and
writes ``real_summary.json`` (``RunLayout.real_summary_json``), which it returns.

**Tools and arms.** The shared arm runs ``igblast``, ``alignair`` (the released image, 3.0.0) and ``alignair-api``
(the image's checkpoint through AlignAIR's public API, ``runners.alignair_api``) on the donor's preprocessed reads;
the genotype arm runs ``igblast`` and ``alignair-api`` restricted to ``real.genotype.arm_alleles``. The comparisons
(``PAIRS``) are igblast with alignair on the shared arm and igblast with alignair-api on both; each tool's genotype arm
against its shared arm (the arm delta) is informational. Every tool and arm is classified against the whole shared
reference with one ``SupportIndex`` per donor and the same truth per read.

**Refusals**, each before any work, with nothing written (``check_config``, then ``run_real``): no donor, an unknown
donor, a test donor without ``split="test"`` (or a dev donor in a test run), an unknown tool, a cap or thread count out
of range, a missing data root, ``igblast`` without its release directory, ``alignair-api`` without its checkpoint
file, an AlignAIR image not pinned by digest; then a cohort file of the sensitivity analysis missing or changed
(``real.fetch.cohort_files``); then Docker missing, or a pinned image whose digest is not the pinned one
(``real_steps.check_images``: sra-tools, samtools, Immcantation and the AlignAIR image); then a run directory made
for another configuration, unless ``force``.

**The manifest** (``real_manifest``) records the configuration and its ``identity``: the donors, split, cap, tools,
reference, preprocessing options, pinned images and every tool's ``fingerprint()``. ``run_real`` refuses a run
directory whose manifest has another identity, naming the fields that differ (a tool's setting as
``<tool>.<key>``), unless ``force``; forced, the manifest is replaced (``replaced``: the old hash and the fields) and
each step redoes what the change affects, every step being skipped only while its own record is current.

**A donor that cannot be scored** is stopped and recorded with its step, reason and evidence, and the others go on:
a fetch that fails verification (``FetchError``), a run the classifier refuses or whose assembly loss is over the
line (``PreprocessError``; ``accept_assembly_loss`` lets it through), or a container step that fails. A tool run that
fails is recorded with its cause and left out of the comparisons.

**The summary** holds the configuration and provenance (images, tool fingerprints, reference, the fetch quality per
run), the sensitivity analysis's genes, per donor (``donors``) its steps' status and seconds, preprocessing record,
genotype, truth yields, oracle self-check, runs (with AlignAIR's AIRR assembly counts), each run's own R1 (``tools``),
the comparisons and arm deltas, and the same pooled over the donors scored (``pooled``), with the stopped donors,
failed runs, and failed and unchecked self-checks listed. The oracle self-check (``real.metrics.oracle_self_check``)
is ``passed``, ``failed`` (the truth contradicts itself) or ``unchecked`` (no oracle read scored: on a small cap the
gene support thresholds exclude every known gene) (``oracle_status``).
"""
from __future__ import annotations

import dataclasses
import os
import platform
import sys
import time
from collections import Counter
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.container import ContainerError, pinned_repo_digest
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import git_state, sha256_file
from alignair_assay.core.reference import DEFAULT_REFERENCE, Reference
from alignair_assay.real.fetch import (
    IGENOTYPER_COMMIT,
    PILOT_DONORS,
    PILOT_SPLIT,
    SRA_TOOLS_IMAGE,
    VDJBASE_SAMPLE,
    FetchError,
)
from alignair_assay.real.genotype import IuisMap, RealGenotype, arm_alleles, merge_identical
from alignair_assay.real.locus import SAMTOOLS_IMAGE
from alignair_assay.real.metrics import SupportIndex, gene_support, oracle_self_check
from alignair_assay.real.preprocess import CAP, IMMCANTATION_IMAGE, PreprocessError
from alignair_assay.real.truth import RealTruth
from alignair_assay.runners import make_runner
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL, host_cache_dir
from alignair_assay.runners.base import Runner, RunnerOptions, RunResult, absolute_path
from alignair_assay.workflow import real_score, real_steps

TOOLS = ("igblast", "alignair", "alignair-api")
GENOTYPE_TOOLS = ("igblast", "alignair-api")
ARMS = ("shared", "genotype")
PAIRS = (("igblast", "alignair", "shared"), ("igblast", "alignair-api", "shared"),
         ("igblast", "alignair-api", "genotype"))
SPLITS = ("dev", "test")
MANIFEST_KIND = "alignair_assay.real.manifest"
SUMMARY_KIND = "alignair_assay.real.summary"
VERSION = "1"
# A donor whose documented known answers are fewer than this share of its reads is flagged (design §10.6).
KNOWN_YIELD_FLAG = 0.05
ASSEMBLY_COUNTS = ("written_records", "complete_assemblies", "partial_assemblies", "failed_assemblies")


@dataclass(frozen=True)
class RealConfig:
    """One ``real`` run: the pilot ``donors`` of one ``split``, the ``data_root`` holding the downloads, the ``cap`` of
    preprocessed sequences per donor, the ``tools`` (of ``TOOLS``), the tool and pRESTO ``threads``, the torch
    ``device`` of alignair-api, the paper's multiplex length filter, whether a run over the assembly's acceptance line
    is let through, the shared ``reference`` and the tool options (``runners.base.RunnerOptions``)."""

    donors: tuple[str, ...]
    split: str
    data_root: str
    cap: int = CAP
    tools: tuple[str, ...] = TOOLS
    threads: int = 8
    device: str = "cpu"
    paper_length_filter: bool = False
    accept_assembly_loss: bool = False
    reference: str = DEFAULT_REFERENCE
    igblast_dir: str | None = None
    alignair_image: str | None = None
    alignair_model: str = DEFAULT_MODEL
    alignair_checkpoint: str | None = None
    batch_size: int = 64


@dataclass
class _Context:
    """What every donor shares: the reference, IUIS map and identical alleles, the runners and their fingerprints,
    which runners are prepared, and the sensitivity analysis's genes."""

    reference: Reference
    runners: dict[str, Runner]
    settings: dict[str, dict]
    setup: RunLayout
    sensitivity: dict
    merged: dict = field(default_factory=dict)
    iuis: IuisMap | None = None
    prepared: set = field(default_factory=set)

    def iuis_for(self, ogrdb_json) -> IuisMap:
        if self.iuis is None:
            self.iuis = real_steps.load_iuis(ogrdb_json, self.reference)
        return self.iuis

    def prepare(self, tool: str) -> None:
        if tool not in self.prepared:
            self.runners[tool].prepare(self.setup, self.reference)
            self.prepared.add(tool)


def default_checkpoint(model: str = DEFAULT_MODEL) -> str:
    """The user's cached file of registry model ``id@version`` (the image's model): alignair-api's checkpoint."""
    model_id, _, version = model.partition("@")
    return str(host_cache_dir() / "models" / model_id / f"{version}.alignair")


def pinned_images(cfg: RealConfig) -> dict[str, str]:
    """The pinned images the run uses, by name: sra-tools, samtools, Immcantation and, with alignair, its image."""
    images = {"sra_tools": SRA_TOOLS_IMAGE, "samtools": SAMTOOLS_IMAGE, "immcantation": IMMCANTATION_IMAGE}
    if "alignair" in cfg.tools:
        images["alignair"] = cfg.alignair_image or DEFAULT_IMAGE
    return images


def check_config(cfg: RealConfig) -> None:
    """ValueError or FileNotFoundError for a configuration that cannot run (the module docstring's first refusals);
    nothing is read beyond the paths' existence, and nothing is written."""
    if not cfg.donors:
        raise ValueError("no donor to run: give --donors")
    if cfg.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {cfg.split!r}")
    for donor in cfg.donors:
        if donor not in PILOT_DONORS:
            raise ValueError(f"unknown donor {donor}: the pilot donors are {', '.join(PILOT_SPLIT['dev'])} (dev) and "
                             f"{', '.join(PILOT_SPLIT['test'])} (test)")
        split = "dev" if donor in PILOT_SPLIT["dev"] else "test"
        if split == "test" and cfg.split != "test":
            raise ValueError(f"{donor} is a test donor: the test donors are locked for claims; pass --split test to "
                             "run them (and only them)")
        if split == "dev" and cfg.split == "test":
            raise ValueError(f"{donor} is a dev donor: a --split test run takes the test donors only "
                             f"({', '.join(PILOT_SPLIT['test'])})")
    if len(set(cfg.donors)) != len(cfg.donors):
        raise ValueError(f"a donor is given twice: {', '.join(cfg.donors)}")
    unknown = [tool for tool in cfg.tools if tool not in TOOLS]
    if unknown or not cfg.tools:
        raise ValueError(f"unknown tool {unknown[0] if unknown else '(none)'!r}: real runs {', '.join(TOOLS)}")
    if isinstance(cfg.cap, bool) or not isinstance(cfg.cap, int) or cfg.cap < 1:
        raise ValueError(f"the cap must be a positive number of sequences, got {cfg.cap!r}")
    if not 1 <= cfg.threads <= (os.cpu_count() or 1) or cfg.batch_size < 1:
        raise ValueError(f"threads must be from 1 to the host's {os.cpu_count()} CPUs and the batch size at least 1, "
                         f"got {cfg.threads} and {cfg.batch_size}")
    if not Path(cfg.data_root).is_dir():
        raise FileNotFoundError(f"no data root {cfg.data_root}: --data-root is the directory of the PRJNA555323 "
                                "downloads")
    if "igblast" in cfg.tools:
        if not cfg.igblast_dir:
            raise ValueError("the igblast runner needs --igblast-dir, the directory of an unpacked IgBLAST release")
        if not Path(cfg.igblast_dir).is_dir():
            raise FileNotFoundError(f"no IgBLAST release directory {cfg.igblast_dir} (--igblast-dir)")
    if "alignair-api" in cfg.tools:
        if not cfg.alignair_checkpoint:
            raise ValueError("alignair-api needs its .alignair checkpoint: --alignair-checkpoint")
        if not Path(cfg.alignair_checkpoint).expanduser().is_file():
            raise FileNotFoundError(f"no AlignAIR checkpoint {cfg.alignair_checkpoint}: alignair-api runs it; fetch it "
                                    f"with `alignair models get {cfg.alignair_model}` or give --alignair-checkpoint")
    if "alignair" in cfg.tools:
        image = cfg.alignair_image or DEFAULT_IMAGE
        try:
            pinned_repo_digest(image)
        except ValueError:
            raise ValueError(f"the AlignAIR image {image} is not pinned by digest (<repository>:<tag>@sha256:...); "
                             "every container the real workflow runs is") from None


def run_real(cfg: RealConfig, layout: RunLayout, *, force: bool = False) -> dict:
    """Run ``cfg`` in ``layout`` as the module docstring describes; returns the summary it writes."""
    check_config(cfg)
    layout = RunLayout(absolute_path(layout.root))
    sensitivity = real_steps.sensitivity_genes(cfg.data_root)
    images = real_steps.check_images(pinned_images(cfg))
    reference = real_steps.load_reference(cfg.reference)
    setup = RunLayout(layout.real_setup_dir())
    options = RunnerOptions(igblast_dir=cfg.igblast_dir, threads=cfg.threads,
                            alignair_image=cfg.alignair_image or DEFAULT_IMAGE, alignair_model=cfg.alignair_model,
                            alignair_checkpoint=cfg.alignair_checkpoint, device=cfg.device,
                            batch_size=cfg.batch_size, cache_dir=str(setup.work_dir("alignair") / "cache"))
    runners = {}
    for tool in cfg.tools:
        runners[tool] = make_runner(tool, options)
        if runners[tool].name != tool:
            raise ValueError(f"the runner registered as {tool!r} is named {runners[tool].name!r}")
    settings = {tool: runner.fingerprint() for tool, runner in runners.items()}
    identity = {"donors": list(cfg.donors), "split": cfg.split, "cap": cfg.cap, "tools": list(cfg.tools),
                "reference": cfg.reference, "paper_length_filter": cfg.paper_length_filter,
                "accept_assembly_loss": cfg.accept_assembly_loss, "images": pinned_images(cfg), "settings": settings}
    config_hash = _manifest(cfg, identity, layout, force=force)
    ctx = _Context(reference=reference, runners=runners, settings=settings, setup=setup, sensitivity=sensitivity,
                   merged=merge_identical(reference))
    blocks, scored = {}, {}
    for donor in cfg.donors:
        blocks[donor], data = _donor(cfg, donor, layout, ctx)
        if data is not None:
            scored[donor] = data
    fetched = real_steps.fetch_manifest(cfg.data_root)
    summary = {
        "kind": SUMMARY_KIND, "version": VERSION, "config": dataclasses.asdict(cfg), "config_hash": config_hash,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "provenance": {"git": git_state(Path(__file__).resolve().parent), "images": images, "tools": settings,
                       "reference": {"name": cfg.reference, "digest": reference.digest()},
                       "ogrdb": None if ctx.iuis is None else ctx.iuis.source,
                       "igenotyper_commit": IGENOTYPER_COMMIT, "fetch_quality": (fetched or {}).get("quality")},
        "sensitivity": sensitivity, "donors": blocks, "pooled": _pooled(cfg, scored, blocks, sensitivity),
        "stopped_donors": [{"donor": donor, "step": block["stopped"]["step"], "reason": block["stopped"]["reason"]}
                           for donor, block in blocks.items() if block["status"] == "stopped"],
        "failed_runs": [{"donor": donor, "tool": tool, "arm": arm, "error": run["error"], "note": run["note"]}
                        for donor, block in blocks.items() for arm, runs in block.get("runs", {}).items()
                        for tool, run in runs.items() if run["status"] != "ok"],
        "oracle_failures": [donor for donor, block in blocks.items()
                            if block["status"] == "done" and block["oracle_self_check"]["status"] == "failed"],
        "oracle_unchecked": [donor for donor, block in blocks.items()
                             if block["status"] == "done" and block["oracle_self_check"]["status"] == "unchecked"],
    }
    real_steps.write_json(layout.real_summary_json(), summary)
    return summary


def _donor(cfg: RealConfig, donor: str, layout: RunLayout,
           ctx: _Context) -> tuple[dict, real_score.DonorScoring | None]:
    """One donor's steps, runs, scores and comparisons, as its summary block, and what pooling needs (None for a
    stopped donor)."""
    block: dict = {"donor": donor, "run": PILOT_DONORS[donor], "sample": VDJBASE_SAMPLE[donor], "status": "done",
                   "stopped": None, "steps": {}}
    steps = block["steps"]
    step = "fetch"
    try:
        fetched, steps["fetch"] = real_steps.step_fetch(donor, cfg.data_root, layout, threads=cfg.threads)
        _say(donor, "fetch", steps["fetch"])
        block["fetch"] = {"quality": fetched.quality, "md5": fetched.md5,
                          "source": (fetched.record.get("reads") or {}).get("detail", {}).get("source")}
        iuis = ctx.iuis_for(fetched.files["ogrdb"])
        step = "genotype"
        genotype, steps["genotype"] = real_steps.step_genotype(donor, fetched, layout, ctx.reference, iuis)
        _say(donor, "genotype", steps["genotype"])
        step = "preprocess"
        record, steps["preprocess"] = real_steps.step_preprocess(
            donor, fetched, layout, cap=cfg.cap, paper_length_filter=cfg.paper_length_filter,
            accept_assembly_loss=cfg.accept_assembly_loss, nproc=cfg.threads)
        _say(donor, "preprocess", steps["preprocess"])
    except (FetchError, PreprocessError, ContainerError) as error:
        block.update(status="stopped", stopped={"step": step, "reason": str(error),
                                                "evidence": getattr(error, "evidence", None)})
        if step == "preprocess":
            block["preprocess"] = _preprocess_block(real_steps.read_json(layout.real_preprocess_json(donor)) or {})
        print(f"{donor} {step}: stopped: {error}", file=sys.stderr, flush=True)
        return block, None
    block.update(protocol=record.get("protocol"), preprocess=_preprocess_block(record))
    truths, steps["truth"] = real_steps.step_truth(donor, layout, genotype, ctx.reference)
    _say(donor, "truth", steps["truth"])
    block["truth"] = _truth_block(truths)

    alleles = (arm_alleles(genotype, ctx.reference, iuis) if set(GENOTYPE_TOOLS) & set(cfg.tools) else None)
    block["genotype"] = _genotype_block(genotype, alleles, ctx.reference)
    steps["runs"], block["runs"], results = {}, {arm: {} for arm in ARMS}, {}
    for tool, arm in expected_runs(cfg):
        started = time.monotonic()
        restricted = alleles if arm == "genotype" else None
        identity = real_steps.run_identity(tool, layout.real_reads(donor), ctx.settings[tool], alleles=restricted,
                                           genotype_json=layout.real_genotype_json(donor))
        result, status = real_steps.current_run(layout, tool, donor, arm, identity), "reused"
        if result is None:
            ctx.prepare(tool)
            result = real_steps.make_run(ctx.runners[tool], tool, donor, arm, layout, identity, alleles=restricted)
            status = "done"
        block["runs"][arm][tool] = _run_block(result)
        status = status if block["runs"][arm][tool]["status"] == "ok" else "failed"
        steps["runs"][f"{arm}/{tool}"] = {"status": status, "seconds": round(time.monotonic() - started, 2)}
        _say(donor, f"run {tool} {arm}", steps["runs"][f"{arm}/{tool}"])
        results[tool, arm] = result

    index = SupportIndex(genotype, iuis, ctx.merged, ctx.reference, support=gene_support(truths))
    weights = {row["sequence_id"]: int(row["consensus_count"])
               for row in real_steps.read_meta(layout.real_reads_meta(donor))}
    data = real_score.DonorScoring(donor=donor, truths={truth.read_id: truth for truth in truths}, index=index,
                                   weights=weights, sd3=frozenset(ctx.sensitivity["genes"]))
    check = oracle_self_check(truths, index)
    block["oracle_self_check"] = {**check, "status": oracle_status(check)}
    steps["scores"], block["tools"] = {}, {arm: {} for arm in ARMS}
    common = {"truth_sha256": sha256_file(layout.real_truth(donor)),
              "genotype_sha256": sha256_file(layout.real_genotype_json(donor)),
              "iuis_ogrdb_sha256": iuis.source.get("sha256"), "reference_digest": ctx.reference.digest()}
    for (tool, arm), result in results.items():
        if block["runs"][arm][tool]["status"] != "ok":
            continue
        started = time.monotonic()
        inputs = {**common, "predictions_sha256": result.provenance.get("predictions_sha256")}
        matching, status = real_score.step_scores(data, tool, arm, layout, inputs)
        steps["scores"][f"{arm}/{tool}"] = {**status, "seconds": round(time.monotonic() - started, 2)}
        block["tools"][arm][tool] = {**real_score.tool_summary(data, tool, arm), "matching": matching}
    block["comparisons"], block["arm_deltas"] = _comparisons(cfg, data, layout, ctx.sensitivity["genes"])
    return block, data


def oracle_status(check: dict) -> str:
    """The verdict of a ``real.metrics.oracle_self_check``: ``passed``; ``failed`` when a known answer's top-1 or gene
    is wrong, or a scored oracle read is not supported (the truth contradicts itself); ``unchecked`` when no oracle
    read is scored (no known answer, or every one excluded, e.g. by the gene support thresholds on a small cap)."""
    if check.get("passed"):
        return "passed"
    known_right = all(check.get(key) in (None, 1.0) for key in ("known_top1", "known_gene_top1"))
    return "unchecked" if known_right and not check.get("scored") else "failed"


def expected_runs(cfg: RealConfig) -> list[tuple[str, str]]:
    """The ``(tool, arm)`` runs of ``cfg``: every tool on the shared arm, then the genotype arm's tools."""
    return ([(tool, "shared") for tool in TOOLS if tool in cfg.tools]
            + [(tool, "genotype") for tool in GENOTYPE_TOOLS if tool in cfg.tools])


def _comparisons(cfg: RealConfig, data: real_score.DonorScoring, layout: RunLayout,
                 genes: list[str]) -> tuple[list[dict], list[dict]]:
    """The donor's comparisons (``PAIRS`` whose tools are in ``cfg``) and arm deltas, each also written to
    ``real_comparison_json``; a pair with a run not scored is listed as not compared, with the reason."""
    comparisons, deltas = [], []
    for a, b, arm in PAIRS:
        if a not in cfg.tools or b not in cfg.tools:
            continue
        entry = {"a": a, "b": b, "arm": arm}
        missing = [tool for tool in (a, b) if (tool, arm) not in data.calls]
        if missing:
            comparisons.append({**entry, "status": "not compared",
                                "reason": f"{', '.join(missing)} on the {arm} arm did not run successfully"})
            continue
        entry.update(status="compared", **real_score.comparison(real_score.pair_scores(data, (a, arm), (b, arm)),
                                                                sensitivity=genes))
        real_score.write_comparison(layout.real_comparison_json(data.donor, arm, f"{a}_vs_{b}"), entry)
        comparisons.append(entry)
    for tool in GENOTYPE_TOOLS:
        if (tool, "shared") in data.calls and (tool, "genotype") in data.calls:
            pair = real_score.pair_scores(data, (tool, "shared"), (tool, "genotype"))
            entry = {"tool": tool, "a": "shared", "b": "genotype", "informational": True,
                     **real_score.comparison(pair, strata=False)}
            real_score.write_comparison(layout.real_comparison_json(data.donor, "arms", tool), entry)
            deltas.append(entry)
    return comparisons, deltas


def _pooled(cfg: RealConfig, scored: dict[str, real_score.DonorScoring], blocks: dict[str, dict],
            sensitivity: dict) -> dict:
    """The comparisons, arm deltas and each run's own R1 pooled over the donors scored (``real_score.pool``)."""
    pooled: dict = {"donors": sorted(scored), "tools": {arm: {} for arm in ARMS}, "comparisons": [], "arm_deltas": []}
    for tool, arm in expected_runs(cfg):
        summaries = [blocks[donor]["tools"][arm][tool] for donor in scored if tool in blocks[donor]["tools"][arm]]
        if summaries:
            pooled["tools"][arm][tool] = real_score.pool_tools(summaries)
    for a, b, arm in PAIRS:
        donors = [donor for donor, data in scored.items() if (a, arm) in data.calls and (b, arm) in data.calls]
        if a in cfg.tools and b in cfg.tools and donors:
            pair = real_score.pool({donor: real_score.pair_scores(scored[donor], (a, arm), (b, arm))
                                    for donor in donors})
            pooled["comparisons"].append({"a": a, "b": b, "arm": arm, "donors": donors, "status": "compared",
                                          **real_score.comparison(pair, sensitivity=sensitivity["genes"])})
    for tool in GENOTYPE_TOOLS:
        donors = [donor for donor, data in scored.items()
                  if (tool, "shared") in data.calls and (tool, "genotype") in data.calls]
        if donors:
            pair = real_score.pool({donor: real_score.pair_scores(scored[donor], (tool, "shared"), (tool, "genotype"))
                                    for donor in donors})
            pooled["arm_deltas"].append({"tool": tool, "a": "shared", "b": "genotype", "informational": True,
                                         "donors": donors, **real_score.comparison(pair, strata=False)})
    return pooled


def _manifest(cfg: RealConfig, identity: dict, layout: RunLayout, *, force: bool) -> str:
    """Write the manifest, keep an identical one, or refuse (ValueError) one made for another configuration; returns
    the configuration's hash."""
    config_hash = real_steps.digest(identity)
    path = layout.real_manifest()
    replaced = None
    if path.is_file():
        old = real_steps.read_json(path) or {}
        if old.get("config_hash") == config_hash:
            return config_hash
        previous, now = _flat(old.get("identity") or {}), _flat(identity)
        changed = [name for name in sorted(previous.keys() | now.keys()) if previous.get(name) != now.get(name)]
        if not force:
            differences = "; ".join(f"{name} {previous.get(name)} -> {now.get(name)}" for name in changed)
            raise ValueError(f"{path} was written for another configuration ({differences}). Rerun with that "
                             "configuration, choose another --run-dir, or pass --force to redo what the change "
                             "affects")
        replaced = {"config_hash": old.get("config_hash"), "changed": changed}
    real_steps.write_json(path, {
        "kind": MANIFEST_KIND, "version": VERSION, "config": dataclasses.asdict(cfg), "identity": identity,
        "config_hash": config_hash, "split": cfg.split,
        "host": {"hostname": platform.node(), "platform": platform.platform(), "python": platform.python_version(),
                 "logical_cpus": os.cpu_count()},
        "git": git_state(Path(__file__).resolve().parent),
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "replaced": replaced})
    return config_hash


def _flat(identity: dict) -> dict:
    """An identity with each tool's setting and each image as its own ``<tool>.<key>`` or ``images.<name>`` field, for
    naming what differs."""
    flat = {name: value for name, value in identity.items() if name not in ("settings", "images")}
    for tool, values in (identity.get("settings") or {}).items():
        flat.update({f"{tool}.{key}": value for key, value in values.items()})
    flat.update({f"images.{name}": image for name, image in (identity.get("images") or {}).items()})
    return flat


def _preprocess_block(record: dict) -> dict:
    """The preprocessing record without its commands: status, protocol, quality, classifier evidence, acceptance,
    counts and pass rates per step, lengths, isotypes and timing."""
    keys = ("status", "protocol", "quality", "quality_steps", "classifier", "accepted", "assembly_acceptance", "counts",
            "pass_rates", "checks", "lengths", "isotypes", "isotypes_unique", "nproc", "wall_seconds", "fingerprint")
    block = {key: record[key] for key in keys if key in record}
    if isinstance(record.get("assembly"), dict):
        block["assembly_failures"] = (record["assembly"].get("fail") or {}).get("kinds")
    block["steps"] = [{key: step.get(key) for key in ("name", "records", "pass_rate", "seconds")}
                      for step in record.get("steps") or ()]
    return block


def _genotype_block(genotype: RealGenotype, alleles: list[str] | None, reference: Reference) -> dict:
    capture = genotype.source.get("capture") or {}
    return {"sample": genotype.source.get("sample"), "v_genes": len(genotype.genes("v")),
            "functional_v_genes": len({a.gene for a in genotype.alleles if a.segment == "v" and a.functional}),
            "d_truth": bool(genotype.genes("d")), "j_truth": bool(genotype.genes("j")),
            "captured": len(genotype.captured or ()), "resolved": len(genotype.resolved or ()),
            "absent_genes": sorted(genotype.absent_genes()), "dropped_d": len(genotype.dropped),
            "arm_v_alleles": None if alleles is None else sum(reference.gene_letter(name) == "v" for name in alleles),
            "capture": {key: capture.get(key) for key in ("spanned_not_carried", "carried_not_spanned",
                                                           "half_resolved")},
            "ogrdb": genotype.source.get("ogrdb")}


def _truth_block(truths: list[RealTruth]) -> dict:
    reads = len(truths)
    documented = sum(truth.known is not None and not truth.novel_known for truth in truths)
    members = Counter(truth.clonotype for truth in truths)
    share = documented / reads if reads else None
    return {"reads": reads, "known": documented, "novel_known": sum(truth.novel_known for truth in truths),
            "known_share": share, "low_known_yield": share is not None and share < KNOWN_YIELD_FLAG,
            "naive": sum(truth.naive for truth in truths),
            "anchor": dict(Counter(truth.anchor or "none" for truth in truths)), "clonotypes": len(members),
            "multi_read_clonotypes": sum(count > 1 for count in members.values()),
            "isotypes": dict(sorted(Counter(truth.isotype for truth in truths).items())),
            "strata": dict(sorted(Counter(real_score.stratum(truth) for truth in truths).items()))}


def _run_block(result: RunResult) -> dict:
    """A run's status (``ok`` only with a return code of 0 and its predictions written), cause, timing and, for
    AlignAIR, its AIRR assembly counts and failure reasons (its run sidecar's)."""
    provenance = result.provenance
    sidecar = provenance.get("sidecar") or {}
    counts = sidecar.get("counts") or {}
    assembly = {key: counts[key] for key in ASSEMBLY_COUNTS if key in counts}
    if assembly and sidecar.get("airr_assembly_fail_reasons"):
        assembly["fail_reasons"] = sidecar["airr_assembly_fail_reasons"]
    ok = result.returncode == 0 and bool(provenance.get("predictions_sha256"))
    error = provenance.get("error") or (None if ok or result.returncode else "the run wrote no predictions")
    return {"status": "ok" if ok else "failed", "returncode": result.returncode, "error": error,
            "note": provenance.get("note"), "wall_seconds": result.wall_seconds, "n_reads": result.n_reads,
            "peak_rss_kb": result.peak_rss_kb, "assembly": assembly or None,
            "genotype_alleles": (provenance.get("genotype") or {}).get("alleles")}


def _say(donor: str, step: str, status: dict) -> None:
    print(f"{donor} {step}: {status['status']} ({status['seconds']:.1f} s)", flush=True)
