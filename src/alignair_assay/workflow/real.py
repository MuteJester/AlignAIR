"""``real``: the real-data pilot from raw reads to scored, cluster-aware comparisons, resumably (P3a design §6-7, 10).

``run_real(cfg, layout)`` runs each donor of ``cfg`` through the steps of ``workflow.real_steps`` (fetch, genotype and
capture, preprocess, truth, the tools' runs) and scores it with ``workflow.real_score``, then pools the donors and
writes ``real_summary.json`` (``RunLayout.real_summary_json``), which it returns, and its markdown report
``real_report.md`` (``RunLayout.real_report_md``, ``report.real``).

**Tools and arms.** The shared arm runs ``igblast``, ``alignair`` (the released image, 3.0.0) and ``alignair-api``
(the image's checkpoint through AlignAIR's public API, ``runners.alignair_api``) on the donor's preprocessed reads;
the genotype arm runs ``igblast`` and ``alignair-api`` restricted to ``real.genotype.arm_alleles``. ``alignair-api``
may also run as checkpoint variants, ``alignair-api@<label>`` with ``checkpoints[label]`` (``runners.variants``),
each a tool of its own on both arms. The comparisons (``compared_pairs``) are igblast with each AlignAIR tool (on the
shared arm, and on the genotype arm for a genotype-arm tool), then every two tools of one AlignAIR base: two
checkpoints on the same reads (``kind`` ``checkpoint``, with the reads the second fixes and breaks), the model
comparison; with the default tools they are ``PAIRS``. Each tool's genotype arm against its shared arm (the arm
delta) is informational. Every tool and arm is classified against the whole shared
reference with one ``SupportIndex`` per donor and the same truth per read. The R1 primary applies no gene-support
threshold; two sensitivity variants are reported beside it: R1 with Mikelov et al. 2024's gene-support thresholds,
and R1 without the reads called to the SD3-only genes (``workflow.real_score``).

**Refusals**, in this order, each before any work and with nothing written to the run directory:

1. ``check_config``: no donor, an unknown donor, a test donor without ``split="test"`` (or a dev donor in a test
   run), an unknown tool (a checkpoint label that is not one, or on a tool other than alignair-api), a cap or
   thread count out of range, a missing data root, ``igblast`` without its release directory, ``alignair-api``
   without its checkpoint file, a variant whose label has no checkpoint or whose checkpoint file is missing, a
   checkpoint label no tool runs, an AlignAIR image not pinned by digest;
2. Docker missing, or a pinned image whose digest is not the pinned one (``real_steps.check_images``: sra-tools,
   samtools, Immcantation and the AlignAIR image);
3. a run directory made for another configuration, unless ``force``, on every field of the manifest's identity but
   the tools' settings (the reference is loaded for its digest first; nothing else has run);
4. a cohort file of the sensitivity analysis that cannot be fetched from its pinned URL or is not the pinned file
   (``real.fetch.cohort_files``; a fetch writes only under the data root);
5. a tool's settings changed, unless ``force``. **The one exception to "before any work"**: a tool's settings are its
   ``fingerprint()``, so they are compared only once it has run, and the AlignAIR image's fingerprint inspects the
   image (pulling it when missing) and installs its model into the run directory's ``real/_setup`` model cache
   (``AlignAIRImage.install_model``) before this refusal; no other file of the run directory is written.

**The manifest** (``real_manifest``) records the configuration and its ``identity``: the donors, split, cap, tools,
reference (its name, its content digest and its gapped forms' digest, so a dataconfig changed under its name is
named in the refusal), preprocessing options, pinned images and every tool's ``fingerprint()``; and the GenAIRR the
reference was built from (``core.provenance.genairr_provenance``), as ``all`` records it. ``run_real`` refuses a run
directory whose manifest has another identity, naming the fields that differ (a tool's setting as
``<tool>.<key>``), unless ``force``; forced, the manifest is replaced (``replaced``: the old hash and the fields) and
each step redoes what the change affects, every step being skipped only while its own record is current.

**Stopping after preprocessing** (``stop_after="preprocess"``, ``--stop-after preprocess``): each donor is fetched,
genotyped and captured, and preprocessed (the protocol classifier's verdict and evidence included), and nothing else
runs: no truth, no tool, no score. Since nothing is scored, any pilot donor may be given, test donors without
``split="test"`` included, and no tool option or cohort file is needed; each donor's block records its split.
``stop_after`` is part of the manifest's identity, so scoring donors in the same run directory afterwards is another
configuration (``force``), whose preprocessing records are reused.

**A donor that cannot be scored** is stopped and recorded with its step, reason and evidence, and the others go on:
a fetch that fails verification (``FetchError``), a run the classifier refuses or whose assembly loss is over the
line (``PreprocessError``; ``accept_assembly_loss`` lets it through), or a container step that fails. A tool run that
fails, or whose predictions hold rows whose ids match no read (its ids drifted), is recorded as failed with its cause
and left out of the comparisons.

**The summary** holds the configuration and provenance (images, tool fingerprints, reference, the fetch quality per
run), the sensitivity analysis's genes, per donor (``donors``) its steps' status and seconds, preprocessing record,
genotype, truth yields, the truth's audit (``truth_health``), oracle self-check, runs (with AlignAIR's AIRR
assembly counts), each run's own R1 (``tools``), the comparisons and arm deltas, and the same pooled over the donors
scored (``pooled``; and per protocol, ``pooled["protocols"]``), with the stopped donors, failed runs, and failed and
unchecked self-checks listed. The oracle self-check (``real.metrics.oracle_self_check``)
is ``passed``, ``failed`` (the truth contradicts itself) or ``unchecked`` (no oracle read scored: the donor's reads
have no documented known answer, or every one is excluded, e.g. ``not_captured``) (``oracle_status``).
"""
from __future__ import annotations

import dataclasses
import os
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from alignair_assay.core.container import ContainerError, pinned_repo_digest
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.provenance import genairr_provenance, git_state, sha256_file
from alignair_assay.core.records import digest, read_json, write_json, write_text
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
from alignair_assay.real.metrics import SupportIndex, decoy_calls, gene_support, oracle_self_check
from alignair_assay.real.preprocess import CAP, IMMCANTATION_IMAGE, PreprocessError
from alignair_assay.real.truth import RealTruth
from alignair_assay.report.real import real_report_markdown
from alignair_assay.runners import check_labels_used, make_runner, parse_tool
from alignair_assay.runners.alignair import DEFAULT_IMAGE, DEFAULT_MODEL, host_cache_dir
from alignair_assay.runners.base import Runner, RunnerOptions, RunResult, absolute_path
from alignair_assay.runners.variants import variant_checkpoint
from alignair_assay.workflow import real_score, real_steps
from alignair_assay.workflow.manifest import changed_fields, describe_changes, manifest_record, tool_order_warnings

TOOLS = ("igblast", "alignair", "alignair-api")
GENOTYPE_TOOLS = ("igblast", "alignair-api")
# The tools that may run as checkpoint variants here (``<tool>@<label>``).
VARIANT_TOOLS = ("alignair-api",)
ARMS = ("shared", "genotype")
# The comparisons of the default tools (``compared_pairs(TOOLS)``).
PAIRS = (("igblast", "alignair", "shared"), ("igblast", "alignair-api", "shared"),
         ("igblast", "alignair-api", "genotype"))
# The ``kind`` of a comparison of two checkpoints of one AlignAIR tool.
CHECKPOINT_PAIR = "checkpoint"
SPLITS = ("dev", "test")
# The steps a run may stop after (``RealConfig.stop_after``).
STOP_AFTER = ("preprocess",)
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
    is let through, the shared ``reference``, the tool options (``runners.base.RunnerOptions``), the step to
    ``stop_after`` (None: every step; ``"preprocess"``: fetch, genotype and preprocess only) and the
    ``checkpoints`` of the checkpoint variants among the tools (label -> ``.alignair`` file)."""

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
    stop_after: str | None = None
    checkpoints: Mapping[str, str] = field(default_factory=dict, hash=False)


@dataclass
class _Context:
    """What every donor shares: the reference, IUIS map and identical alleles, the runners and their fingerprints,
    which runners are prepared, and the sensitivity analysis's genes."""

    reference: Reference
    runners: dict[str, Runner]
    settings: dict[str, dict]
    setup: RunLayout
    sensitivity: dict | None
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
    """The pinned images the run uses, by name: sra-tools, samtools, Immcantation and, with alignair run (not when the
    run stops after preprocessing), its image."""
    images = {"sra_tools": SRA_TOOLS_IMAGE, "samtools": SAMTOOLS_IMAGE, "immcantation": IMMCANTATION_IMAGE}
    if "alignair" in cfg.tools and cfg.stop_after is None:
        images["alignair"] = cfg.alignair_image or DEFAULT_IMAGE
    return images


def check_config(cfg: RealConfig) -> None:
    """ValueError or FileNotFoundError for a configuration that cannot run (the module docstring's first refusals);
    nothing is read beyond the paths' existence, and nothing is written. A run that stops after preprocessing scores
    nothing, so it takes any pilot donor and needs no tool option."""
    if not cfg.donors:
        raise ValueError("no donor to run: give --donors")
    if cfg.split not in SPLITS:
        raise ValueError(f"split must be one of {', '.join(SPLITS)}, got {cfg.split!r}")
    if cfg.stop_after not in (None, *STOP_AFTER):
        raise ValueError(f"stop_after must be None or one of {', '.join(STOP_AFTER)}, got {cfg.stop_after!r}")
    for donor in cfg.donors:
        if donor not in PILOT_DONORS:
            raise ValueError(f"unknown donor {donor}: the pilot donors are {', '.join(PILOT_SPLIT['dev'])} (dev) and "
                             f"{', '.join(PILOT_SPLIT['test'])} (test)")
        if cfg.stop_after is not None:
            continue  # nothing is scored: any pilot donor, its split recorded
        split = donor_split(donor)
        if split == "test" and cfg.split != "test":
            raise ValueError(f"{donor} is a test donor: the test donors are locked for claims; pass --split test to "
                             "run them (and only them)")
        if split == "dev" and cfg.split == "test":
            raise ValueError(f"{donor} is a dev donor: a --split test run takes the test donors only "
                             f"({', '.join(PILOT_SPLIT['test'])})")
    if len(set(cfg.donors)) != len(cfg.donors):
        raise ValueError(f"a donor is given twice: {', '.join(cfg.donors)}")
    if not cfg.tools:
        raise ValueError(f"unknown tool '(none)': real runs {', '.join(TOOLS)}")
    for tool in cfg.tools:
        base, label = parse_tool(tool)  # a label that is not one, or on a tool without checkpoints: ValueError
        if base not in TOOLS or (label is not None and base not in VARIANT_TOOLS):
            raise ValueError(f"unknown tool {tool!r}: real runs {', '.join(TOOLS)} and the checkpoint variants "
                             f"{', '.join(f'{name}@<label>' for name in VARIANT_TOOLS)} (--checkpoint <label>=PATH)")
    if isinstance(cfg.cap, bool) or not isinstance(cfg.cap, int) or cfg.cap < 1:
        raise ValueError(f"the cap must be a positive number of sequences, got {cfg.cap!r}")
    if not 1 <= cfg.threads <= (os.cpu_count() or 1) or cfg.batch_size < 1:
        raise ValueError(f"threads must be from 1 to the host's {os.cpu_count()} CPUs and the batch size at least 1, "
                         f"got {cfg.threads} and {cfg.batch_size}")
    if not Path(cfg.data_root).is_dir():
        raise FileNotFoundError(f"no data root {cfg.data_root}: --data-root is the directory of the PRJNA555323 "
                                "downloads")
    if cfg.stop_after is not None:
        return  # no tool runs
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
    for tool in cfg.tools:
        base, label = parse_tool(tool)
        if label is not None:
            variant_checkpoint(label, cfg.checkpoints, tool=tool)
    check_labels_used(cfg.tools, cfg.checkpoints)
    if "alignair" in cfg.tools:
        image = cfg.alignair_image or DEFAULT_IMAGE
        try:
            pinned_repo_digest(image)
        except ValueError:
            raise ValueError(f"the AlignAIR image {image} is not pinned by digest (<repository>:<tag>@sha256:...); "
                             "every container the real workflow runs is") from None


def manifest_identity(cfg: RealConfig, reference: Reference, settings: dict[str, dict] | None = None) -> dict:
    """The manifest's ``identity`` (the module docstring): the donors, split, cap, tools (none when the run stops
    after preprocessing), the reference's name and its ``digest()`` and ``gapped_digest()``, the preprocessing
    options, ``stop_after`` and the pinned images; and, given, the tools' ``settings`` (each one's
    ``fingerprint()``), which ``run_real`` adds once it has them."""
    scoring = cfg.stop_after is None
    identity = {"donors": list(cfg.donors), "split": cfg.split, "cap": cfg.cap,
                "tools": list(cfg.tools) if scoring else [], "reference": cfg.reference,
                "reference_digest": reference.digest(), "reference_gapped_digest": reference.gapped_digest(),
                "paper_length_filter": cfg.paper_length_filter, "accept_assembly_loss": cfg.accept_assembly_loss,
                "stop_after": cfg.stop_after, "images": pinned_images(cfg)}
    if settings is not None:
        identity["settings"] = settings
    return identity


def donor_split(donor: str) -> str:
    """The pilot split of ``donor``: ``dev`` or ``test`` (``real.fetch.PILOT_SPLIT``)."""
    return "dev" if donor in PILOT_SPLIT["dev"] else "test"


def run_real(cfg: RealConfig, layout: RunLayout, *, force: bool = False) -> dict:
    """Run ``cfg`` in ``layout`` as the module docstring describes; returns the summary it writes. A baseline given
    second is warned of first (``workflow.manifest.tool_order_warnings``)."""
    check_config(cfg)
    for warning in tool_order_warnings(cfg.tools):
        print(f"warning: {warning}", file=sys.stderr)
    layout = RunLayout(absolute_path(layout.root))
    scoring = cfg.stop_after is None
    images = real_steps.check_images(pinned_images(cfg))
    reference = real_steps.load_reference(cfg.reference)
    identity = manifest_identity(cfg, reference)
    _check_manifest(identity, layout, force=force, ignore=("settings",))  # before any work: refusal 3
    sensitivity = real_steps.sensitivity_genes(cfg.data_root) if scoring else None
    setup = RunLayout(layout.real_setup_dir())
    options = RunnerOptions(igblast_dir=cfg.igblast_dir, threads=cfg.threads,
                            alignair_image=cfg.alignair_image or DEFAULT_IMAGE, alignair_model=cfg.alignair_model,
                            alignair_checkpoint=cfg.alignair_checkpoint, device=cfg.device,
                            batch_size=cfg.batch_size, cache_dir=str(setup.work_dir("alignair") / "cache"),
                            checkpoints=cfg.checkpoints)
    runners = {}
    for tool in cfg.tools if scoring else ():
        runners[tool] = make_runner(tool, options)
        if runners[tool].name != tool:
            raise ValueError(f"the runner registered as {tool!r} is named {runners[tool].name!r}")
    settings = {tool: runner.fingerprint() for tool, runner in runners.items()}  # the image's writes real/_setup
    identity["settings"] = settings
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
        "provenance": {"git": git_state(Path(__file__).resolve().parent), "genairr": genairr_provenance(),
                       "images": images, "tools": settings,
                       "reference": {"name": cfg.reference, "digest": reference.digest(),
                                     "gapped_digest": reference.gapped_digest()},
                       "ogrdb": None if ctx.iuis is None else ctx.iuis.source,
                       "igenotyper_commit": IGENOTYPER_COMMIT, "fetch_quality": (fetched or {}).get("quality")},
        "sensitivity": sensitivity, "donors": blocks, "pooled": _pooled(cfg, scored, blocks, sensitivity),
        "stopped_donors": [{"donor": donor, "step": block["stopped"]["step"], "reason": block["stopped"]["reason"]}
                           for donor, block in blocks.items() if block["status"] == "stopped"],
        "failed_runs": [{"donor": donor, "tool": tool, "arm": arm, "error": run["error"], "note": run["note"]}
                        for donor, block in blocks.items() for arm, runs in block.get("runs", {}).items()
                        for tool, run in runs.items() if run["status"] != "ok"],
        "oracle_failures": [donor for donor, block in blocks.items()
                            if (block.get("oracle_self_check") or {}).get("status") == "failed"],
        "oracle_unchecked": [donor for donor, block in blocks.items()
                             if (block.get("oracle_self_check") or {}).get("status") == "unchecked"],
    }
    write_json(layout.real_summary_json(), summary)
    write_text(layout.real_report_md(), real_report_markdown(summary))
    return summary


def _donor(cfg: RealConfig, donor: str, layout: RunLayout,
           ctx: _Context) -> tuple[dict, real_score.DonorScoring | None]:
    """One donor's steps, runs, scores and comparisons, as its summary block, and what pooling needs (None for a
    stopped donor)."""
    block: dict = {"donor": donor, "split": donor_split(donor), "run": PILOT_DONORS[donor],
                   "sample": VDJBASE_SAMPLE[donor], "status": "done", "stopped": None, "steps": {}}
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
            block["preprocess"] = _preprocess_block(read_json(layout.real_preprocess_json(donor)) or {})
        print(f"{donor} {step}: stopped: {error}", file=sys.stderr, flush=True)
        return block, None
    block.update(protocol=record.get("protocol"), preprocess=_preprocess_block(record))
    if cfg.stop_after == "preprocess":
        block.update(stopped_after="preprocess", genotype=_genotype_block(genotype, None, ctx.reference))
        return block, None
    truths, block["truth_health"], steps["truth"] = real_steps.step_truth(donor, layout, genotype, ctx.reference,
                                                                           iuis)
    _say(donor, "truth", steps["truth"])
    block["truth"] = _truth_block(truths)

    alleles = arm_alleles(genotype, ctx.reference, iuis) if genotype_tools(cfg.tools) else None
    block["genotype"] = _genotype_block(genotype, alleles, ctx.reference)
    steps["runs"], block["runs"], results = {}, {arm: {} for arm in ARMS}, {}
    for tool, arm in expected_runs(cfg):
        started = time.monotonic()
        restricted = alleles if arm == "genotype" else None
        identity = real_steps.run_identity(tool, layout.real_reads(donor), ctx.settings[tool],
                                           reference_digest=ctx.reference.digest(), alleles=restricted,
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

    data = donor_scoring(donor, layout, truths, genotype, iuis, ctx.reference, ctx.merged, ctx.sensitivity["genes"])
    steps["scores"], block["tools"] = {}, {arm: {} for arm in ARMS}
    common = scoring_inputs(donor, layout, iuis, ctx.reference)
    block["oracle_self_check"] = self_check(data, layout, common)
    for (tool, arm), result in results.items():
        if block["runs"][arm][tool]["status"] != "ok":
            continue
        started = time.monotonic()
        inputs = {**common, "predictions_sha256": result.provenance.get("predictions_sha256")}
        matching, status = real_score.step_scores(data, tool, arm, layout, inputs)
        if matching.get("extra") or matching.get("duplicates"):
            # rows whose ids match no read, or repeat one: the ids drifted, so nothing the run predicted can be
            # trusted to be paired with its read (as P1/P2 refuse a prediction file that does not match its truth)
            data.calls.pop((tool, arm), None)
            block["runs"][arm][tool].update(status="failed", error=_drift(matching))
            steps["scores"][f"{arm}/{tool}"] = {"status": "failed", "seconds": round(time.monotonic() - started, 2)}
            continue
        steps["scores"][f"{arm}/{tool}"] = {**status, "seconds": round(time.monotonic() - started, 2)}
        block["tools"][arm][tool] = {**real_score.tool_summary(data, tool, arm), "matching": matching}
    block["comparisons"], block["arm_deltas"] = _comparisons(cfg, data, layout, ctx.sensitivity["genes"])
    return block, data


def donor_scoring(donor: str, layout: RunLayout, truths: list[RealTruth], genotype: RealGenotype, iuis: IuisMap,
                  reference: Reference, merged: dict, sd3_genes: Iterable[str]) -> real_score.DonorScoring:
    """What the donor's reads are scored with (``real_score.DonorScoring``): their ``truths`` (in read order), the
    ``SupportIndex`` of the R1 primary and of the gene-support variant over the gene support of the truths' known
    answers, the reads' consensus counts (``reads.tsv`` in ``layout``) and the SD3 sensitivity analysis's genes.
    ``merged`` is ``real.genotype.merge_identical(reference)``."""
    support = gene_support(truths)
    index = SupportIndex(genotype, iuis, merged, reference, support=support)
    thresholded = SupportIndex(genotype, iuis, merged, reference, support=support, thresholds=True)
    weights = {row["sequence_id"]: int(row["consensus_count"])
               for row in real_steps.read_meta(layout.real_reads_meta(donor))}
    return real_score.DonorScoring(donor=donor, truths={truth.read_id: truth for truth in truths}, index=index,
                                   thresholded=thresholded, weights=weights, sd3=frozenset(sd3_genes))


def scoring_inputs(donor: str, layout: RunLayout, iuis: IuisMap, reference: Reference) -> dict:
    """The digests every run's scores of the donor are keyed on (``real_score.step_scores``' ``inputs`` without the
    run's own ``predictions_sha256``): the truth and genotype files in ``layout``, the IUIS map's OGRDB file and the
    reference's content and gapped forms."""
    return {"truth_sha256": sha256_file(layout.real_truth(donor)),
            "genotype_sha256": sha256_file(layout.real_genotype_json(donor)),
            "iuis_ogrdb_sha256": iuis.source.get("sha256"), "reference_digest": reference.digest(),
            "reference_gapped_digest": reference.gapped_digest()}


def self_check(data: real_score.DonorScoring, layout: RunLayout, common: dict) -> dict:
    """The donor's truth self-check: ``real.metrics.oracle_self_check`` of its truths, the oracle through the tools'
    path (``pipeline``: ``real_score.pipeline_oracle``, whose AIRR TSV and scores it writes in ``layout``, keyed on
    ``common``) and the ``decoys`` (``real.metrics.decoy_calls``), with its ``status`` (``oracle_status``)."""
    check = {**oracle_self_check(list(data.truths.values()), data.index),
             "pipeline": real_score.pipeline_oracle(data, layout, common), "decoys": decoy_calls(data.index)}
    return {**check, "status": oracle_status(check)}


def _drift(matching: dict) -> str:
    """Why a run's rows cannot be paired with its reads: rows whose ids match no read, rows repeating a read's id."""
    found = []
    if matching.get("extra"):
        found.append(f"{matching['extra']} predicted rows match no read (e.g. {', '.join(matching['extra_ids'])})")
    if matching.get("duplicates"):
        count = matching["duplicates"]
        found.append(f"{count} predicted {'row repeats a read' if count == 1 else 'rows repeat a read'}'s id")
    return f"{'; '.join(found)}: the run's ids drifted from the reads it was given, so it is not scored"


def oracle_status(check: dict) -> str:
    """The verdict of a ``real.metrics.oracle_self_check`` with its ``pipeline`` and ``decoys``: ``passed``;
    ``failed`` when a known answer's top-1 or gene is wrong, a scored oracle read is not supported (the truth
    contradicts itself), the pipeline oracle (``real_score.pipeline_oracle``) disagrees with it, or a decoy
    (``real.metrics.decoy_calls``) is not its class; ``unchecked`` when no oracle read is scored: no documented known
    answer, or every one excluded (e.g. ``not_captured``). The R1 primary applies no gene-support threshold, so the
    thresholds never make it unchecked."""
    if (check.get("pipeline") or {}).get("passed") is False or (check.get("decoys") or {}).get("passed") is False:
        return "failed"
    if check.get("passed"):
        return "passed"
    known_right = all(check.get(key) in (None, 1.0) for key in ("known_top1", "known_gene_top1", "ref_known_top1"))
    return "unchecked" if known_right and not check.get("scored") else "failed"


def ordered_tools(tools: Iterable[str]) -> list[str]:
    """``tools`` without repeats, by their base's place in ``TOOLS``; one base's tools (the plain tool and its
    checkpoint variants) in the order given."""
    return sorted(dict.fromkeys(tools), key=lambda tool: TOOLS.index(parse_tool(tool)[0]))


def genotype_tools(tools: Iterable[str]) -> list[str]:
    """The tools of ``tools`` that run on the genotype arm (their base is in ``GENOTYPE_TOOLS``), in order."""
    return [tool for tool in ordered_tools(tools) if parse_tool(tool)[0] in GENOTYPE_TOOLS]


def is_checkpoint_pair(a: str, b: str) -> bool:
    """Two tools of one AlignAIR base: one tool on two checkpoints (a plain tool and a variant, or two variants)."""
    base = parse_tool(a)[0]
    return a != b and base != "igblast" and parse_tool(b)[0] == base


def compared_pairs(tools: Iterable[str]) -> list[tuple[str, str, str]]:
    """The ``(a, b, arm)`` comparisons of ``tools`` (``ordered_tools``): igblast with each AlignAIR tool, on the shared
    arm and, for a genotype-arm tool, on the genotype arm; then every two tools of one AlignAIR base
    (``is_checkpoint_pair``, a first) on each arm they run. With the default ``TOOLS`` they are ``PAIRS``."""
    tools = ordered_tools(tools)
    genotype = set(genotype_tools(tools))

    def arms(tool: str) -> list[str]:
        return ["shared", "genotype"] if tool in genotype else ["shared"]

    found = [("igblast", tool, arm) for tool in tools if "igblast" in tools and tool != "igblast" for arm in arms(tool)]
    found += [(a, b, arm) for i, a in enumerate(tools) for b in tools[i + 1:] if is_checkpoint_pair(a, b)
              for arm in arms(a)]
    return found


def expected_runs(cfg: RealConfig) -> list[tuple[str, str]]:
    """The ``(tool, arm)`` runs of ``cfg``: every tool on the shared arm, then the genotype arm's tools
    (``ordered_tools``)."""
    return ([(tool, "shared") for tool in ordered_tools(cfg.tools)]
            + [(tool, "genotype") for tool in genotype_tools(cfg.tools)])


def _pair_entry(a: str, b: str, arm: str) -> dict:
    """A comparison's head: ``a``, ``b`` and ``arm``, and ``kind`` ``CHECKPOINT_PAIR`` for two checkpoints of one
    tool."""
    return {"a": a, "b": b, "arm": arm, **({"kind": CHECKPOINT_PAIR} if is_checkpoint_pair(a, b) else {})}


def _comparisons(cfg: RealConfig, data: real_score.DonorScoring, layout: RunLayout,
                 genes: list[str]) -> tuple[list[dict], list[dict]]:
    """The donor's comparisons (``compared_pairs``) and arm deltas, each also written to ``real_comparison_json``; a
    pair with a run not scored is listed as not compared, with the reason. A comparison of two checkpoints also
    counts the reads the second fixes and breaks (``real_score.comparison(..., changes=True)``)."""
    comparisons, deltas = [], []
    for a, b, arm in compared_pairs(cfg.tools):
        entry = _pair_entry(a, b, arm)
        missing = [tool for tool in (a, b) if (tool, arm) not in data.calls]
        if missing:
            comparisons.append({**entry, "status": "not compared",
                                "reason": f"{', '.join(missing)} on the {arm} arm did not run successfully"})
            continue
        entry.update(status="compared", **real_score.comparison(
            real_score.pair_scores(data, (a, arm), (b, arm)), sensitivity=genes,
            gene_support=real_score.pair_scores(data, (a, arm), (b, arm), variant="gene_support"),
            clusterings=True, changes=is_checkpoint_pair(a, b), donor=data.donor))
        real_score.write_comparison(layout.real_comparison_json(data.donor, arm, f"{a}_vs_{b}"), entry)
        comparisons.append(entry)
    for tool in genotype_tools(cfg.tools):
        if (tool, "shared") in data.calls and (tool, "genotype") in data.calls:
            pair = real_score.pair_scores(data, (tool, "shared"), (tool, "genotype"))
            entry = {"tool": tool, "a": "shared", "b": "genotype", "informational": True,
                     **real_score.comparison(pair, strata=False)}
            real_score.write_comparison(layout.real_comparison_json(data.donor, "arms", tool), entry)
            deltas.append(entry)
    return comparisons, deltas


def _pooled(cfg: RealConfig, scored: dict[str, real_score.DonorScoring], blocks: dict[str, dict],
            sensitivity: dict) -> dict:
    """The comparisons, arm deltas and each run's own R1 pooled over the donors scored (``real_score.pool``), and the
    comparisons pooled over each protocol's donors (``protocols``: protocol -> its ``donors`` and ``comparisons``),
    where the pre-declared primary is read: the protocols' mix of reads sets the pool over both."""
    pooled: dict = {"donors": sorted(scored), "tools": {arm: {} for arm in ARMS}, "comparisons": [], "arm_deltas": [],
                    "protocols": {}}
    by_protocol: dict[str, list[str]] = {}
    for donor in sorted(scored):
        by_protocol.setdefault(blocks[donor].get("protocol") or "unknown", []).append(donor)
    for protocol, members in sorted(by_protocol.items()):
        comparisons = []
        for a, b, arm in compared_pairs(cfg.tools):
            donors = [donor for donor in members if (a, arm) in scored[donor].calls and (b, arm) in scored[donor].calls]
            if donors:
                pair = real_score.pool({donor: real_score.pair_scores(scored[donor], (a, arm), (b, arm))
                                        for donor in donors})
                comparisons.append({**_pair_entry(a, b, arm), "donors": donors, "status": "compared",
                                    **real_score.comparison(pair, clusterings=True,
                                                            changes=is_checkpoint_pair(a, b))})
        pooled["protocols"][protocol] = {"donors": members, "comparisons": comparisons}
    for tool, arm in expected_runs(cfg):
        summaries = [blocks[donor]["tools"][arm][tool] for donor in scored if tool in blocks[donor]["tools"][arm]]
        if summaries:
            pooled["tools"][arm][tool] = real_score.pool_tools(summaries)
    for a, b, arm in compared_pairs(cfg.tools):
        donors = [donor for donor, data in scored.items() if (a, arm) in data.calls and (b, arm) in data.calls]
        if donors:
            pairs = {variant: real_score.pool({donor: real_score.pair_scores(scored[donor], (a, arm), (b, arm),
                                                                             variant=variant) for donor in donors})
                     for variant in real_score.VARIANTS}
            pooled["comparisons"].append({**_pair_entry(a, b, arm), "donors": donors, "status": "compared",
                                          **real_score.comparison(pairs["primary"], sensitivity=sensitivity["genes"],
                                                                  gene_support=pairs["gene_support"],
                                                                  clusterings=True,
                                                                  changes=is_checkpoint_pair(a, b))})
    for tool in genotype_tools(cfg.tools):
        donors = [donor for donor, data in scored.items()
                  if (tool, "shared") in data.calls and (tool, "genotype") in data.calls]
        if donors:
            pair = real_score.pool({donor: real_score.pair_scores(scored[donor], (tool, "shared"), (tool, "genotype"))
                                    for donor in donors})
            pooled["arm_deltas"].append({"tool": tool, "a": "shared", "b": "genotype", "informational": True,
                                         "donors": donors, **real_score.comparison(pair, strata=False)})
    return pooled


def _check_manifest(identity: dict, layout: RunLayout, *, force: bool, ignore: tuple[str, ...] = ()) -> list | None:
    """The fields of ``identity`` (``workflow.manifest.flat``) that differ from the run directory's manifest, leaving
    out the top-level keys ``ignore`` on both sides; a field one side lacks (a manifest written before the field
    existed) differs and is shown as ``(absent)``. None without a manifest, or when its whole identity's hash is
    ``identity``'s. ValueError naming them when they differ, unless ``force``; without ``ignore``, also when no field
    differs but the hash does ("identity format changed"). Reads only."""
    path = layout.real_manifest()
    if not path.is_file():
        return None
    old = read_json(path) or {}
    if not ignore and old.get("config_hash") == digest(identity):
        return None
    previous = old.get("identity") or {}
    changed = changed_fields(previous, identity, ignore=ignore)
    if force:
        return changed
    if changed:
        raise ValueError(f"{path} was written for another configuration "
                         f"({describe_changes(previous, identity, changed)}). Rerun with that configuration, choose "
                         "another --run-dir, or pass --force to redo what the change affects")
    if not ignore:
        raise ValueError(f"{path} was written for another configuration: no field of its identity differs, but its "
                         "hash does (identity format changed: the workflow that wrote it recorded its configuration "
                         "otherwise). Choose another --run-dir, or pass --force to redo what the change affects")
    return changed


def _manifest(cfg: RealConfig, identity: dict, layout: RunLayout, *, force: bool) -> str:
    """Write the manifest (``workflow.manifest.manifest_record``), keep an identical one, or refuse (ValueError,
    ``_check_manifest``) one made for another configuration; returns the configuration's hash."""
    config_hash = digest(identity)
    path = layout.real_manifest()
    replaced = None
    changed = _check_manifest(identity, layout, force=force)
    if changed is None and path.is_file():
        return config_hash
    if changed is not None:
        old = read_json(path) or {}
        replaced = {"config_hash": old.get("config_hash"), "changed": changed}
    write_json(path, manifest_record(kind=MANIFEST_KIND, version=VERSION, config=dataclasses.asdict(cfg),
                                     identity=identity, config_hash=config_hash, split=cfg.split, replaced=replaced))
    return config_hash


def _preprocess_block(record: dict) -> dict:
    """The preprocessing record without its commands: status, protocol, quality, classifier evidence, acceptance,
    the assembly evidence (``assembly``: the deciles of the pairs that passed and failed, and the failures by kind,
    also as ``assembly_failures``), counts and pass rates per step, lengths, isotypes and timing."""
    keys = ("status", "protocol", "quality", "quality_steps", "classifier", "accepted", "assembly_acceptance", "counts",
            "pass_rates", "checks", "lengths", "isotypes", "isotypes_unique", "nproc", "wall_seconds", "fingerprint")
    block = {key: record[key] for key in keys if key in record}
    if isinstance(record.get("assembly"), dict):
        block["assembly"] = record["assembly"]
        block["assembly_failures"] = (record["assembly"].get("fail") or {}).get("kinds")
    block["steps"] = [{key: step.get(key) for key in ("name", "records", "pass_rate", "seconds")}
                      for step in record.get("steps") or ()]
    return block


def _genotype_block(genotype: RealGenotype, alleles: list[str] | None, reference: Reference) -> dict:
    capture = genotype.source.get("capture") or {}
    return {"sample": genotype.source.get("sample"), "v_genes": len(genotype.genes("v")),
            "functional_v_genes": len({a.gene for a in genotype.alleles if a.segment == "v" and a.functional}),
            "d_truth": bool(genotype.genes("d")), "j_truth": bool(genotype.genes("j")),
            "j_alleles": [{"allele": allele.allele_id, "haplotype": allele.haplotype} for allele in genotype.alleles
                          if allele.segment == "j"],
            "captured": len(genotype.captured or ()), "resolved": len(genotype.resolved or ()),
            "absent_genes": sorted(genotype.absent_genes()), "dropped_d": len(genotype.dropped),
            "arm_v_alleles": None if alleles is None else sum(reference.gene_letter(name) == "v" for name in alleles),
            "capture": {key: capture.get(key) for key in ("spanned_not_carried", "carried_not_spanned", "uncalled",
                                                           "absent", "half_resolved", "unknown_haplotypes")},
            "ogrdb": genotype.source.get("ogrdb")}


def _truth_block(truths: list[RealTruth]) -> dict:
    reads = len(truths)
    documented = sum(truth.known is not None and not truth.novel_known for truth in truths)
    referenced = sum(truth.ref_known is not None for truth in truths)
    members = Counter(truth.clonotype for truth in truths)
    share = documented / reads if reads else None
    return {"reads": reads, "known": documented, "novel_known": sum(truth.novel_known for truth in truths),
            "known_share": share, "low_known_yield": share is not None and share < KNOWN_YIELD_FLAG,
            "ref_known": referenced, "ref_known_share": referenced / reads if reads else None,
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
