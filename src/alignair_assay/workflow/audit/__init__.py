"""``audit``: the readiness gate for model comparisons.

``run_audit(cfg, layout)`` proves, before any AlignAIR model work, that the assay can be trusted to compare two
checkpoints: the truth checks itself, its outputs reproduce, a model compared with itself differs by exactly nothing
on every surface (``all``, ``diagnose``, ``real``), and IgBLAST lands where independent evidence says it should. It
writes ``layout.audit_json`` and ``layout.audit_md`` (``report.audit``), whose first line is the verdict.

**Checks.** Each is ``{name, status, n, evidence, reason}``: ``status`` is ``pass``, ``fail``, ``warn`` (worth a
look, never a failure) or ``skipped`` (not checkable now, with why); ``n`` is how many reads or cases it rests on;
``evidence`` holds the numbers, its ``summary`` a one-line digest of them. A check that passes on nothing (``n`` 0)
fails (``settle``), and so does a check that raises (its error is the reason). The ``INFORMATIONAL`` checks warn at
worst; every other one gates (``GATING``: it can fail the audit). The audit passes when no check fails, no gating
check was skipped and at least one passed (``verdict``); a skipped gating check makes it ``incomplete`` (exit 3),
since what was not checked may have failed, and an audit that checked nothing never passes.

- Self-checks:
  - ``self_check.sim_oracle``: the stored ``all`` run's oracle is perfect on every suite (``workflow.summary
    .oracle_check`` of its stored oracle scores, read only);
  - ``self_check.sim_oracle_current``: the oracle of the null ``all`` run below, on a fresh dev build at this code;
  - ``self_check.sim_staleness``: the stored run's suite builds against this checkout's suite, truth and GenAIRR
    code (``workflow.build.build_changes``); informational, ``warn`` when they moved;
  - ``self_check.real_oracle``: every dev donor of the stored ``real`` run passed its oracle self-check (the pipeline
    path and the decoys included);
  - ``self_check.real_truth_health``: the truth-health flags of those donors, ``warn`` when there are any.
- Reproducibility (in scratch):
  - ``reproducible.sim_build``: ``REPRO_SUITE`` built twice at ``REPRO_SCALE`` on the dev split; its cases, truth and
    reads must be byte-identical. Compared only when GenAIRR's source did not move during the two builds
    (``genairr_digest`` before, between and after); otherwise ``skipped (GenAIRR moved)``, never a failure;
  - ``reproducible.real_scores``: one dev donor's truth and every run's scores recomputed from the stored reads,
    genotype and predictions, and its self-check redone; each file's content (its rows, and its header without the
    code digest and the fingerprint that holds it) must equal the stored one's (``compare_records``). Its bytes are
    reported, not required: files written before JSON-lines gzip was reproducible carry a time in their gzip header,
    and the scores' inputs name the truth file by those bytes, so a stored truth's digest stands for the recomputed
    one's when the two hold the same content.
- Null comparisons (one checkpoint under ``NULL_LABELS``), the simulated ones on one shared dev build of
  ``null_suites`` and ``diagnose_suites`` at ``null_scale`` (``layout.audit_scratch("null_sim")``): every Δ 0, no
  significant verdict, no read won or lost
  (fixed = broken = 0), each side's own scores equal and every read either side scores paired (``compare_problems``);
  the comparison must pair something (``pairing``): a comparison that pairs nothing is not null, it is empty; and it
  must hold everything expected of it, since what is missing is never null: every end-to-end headline metric
  (``SIM_HEADLINE``, ``REAL_HEADLINE``) on every suite and arm, and in ``diagnose`` every suite diagnosed, each with
  its head family (one metric per ``HEAD_BOUNDARY_PREFIXES`` at least) and its scored reads. Its ``n`` is the fewest
  reads a headline metric pairs:
  - ``null.all``: ``all --tools alignair-local@a,alignair-local@b`` on ``NULL_SUITES`` (S7's genotype arm among
    them: the genotype path of ``all`` with checkpoint variants);
  - ``null.diagnose``: the checkpoint file against itself loaded in memory (two model keys, each diagnosed) on
    ``DIAGNOSE_SUITES``, built where ``all`` did not already build them;
  - ``null.real``: ``real --tools alignair-api@a,alignair-api@b --cap REAL_CAP`` on ``REAL_DONOR``.
- The positive control, ``positive.diagnose``: the checkpoint with ``POSITIVE_MODULE`` zeroed
  (``diagnostics.perturb.zero_module``, written to ``layout.audit_scratch("positive")``) diagnosed against the
  checkpoint on the first suite both null surfaces share (S1): the perturbed head metric ``POSITIVE_METRIC`` must be
  worse (Δ < 0, the verdict "baseline better"), with reads broken and fewer fixed, and the suite detail must render
  that direction: a comparison that cannot see a broken head, or reads it backwards, is no instrument. The metric's
  rate on the paired reads, the checkpoint's and the perturbed model's, is in the evidence: the room it had to get
  worse (a metric the checkpoint almost never gets right cannot show a broken head, and the check then fails).
- Cross-surface agreement, ``agreement.all_vs_diagnose``: on that suite's one build, ``all``'s per-case end-to-end
  scores of ``alignair-local@b`` must equal ``diagnose``'s per-case baseline scores of the same checkpoint (its file's
  model key), case by case, on every metric both score: the CLI's AIRR output and the in-process replay are one
  pipeline.
- IgBLAST bands:
  - ``band.real_known``: IgBLAST's ``real.v.known_top1`` and ``real.v.ref_known_top1`` on the shared arm at least
    ``KNOWN_FLOOR`` on every dev donor of the stored ``real`` run;
  - ``band.sim_v_top1_current``: its S1 ``v.top1_in_set`` at this code, IgBLAST run now on a fresh dev S1, within
    ``V_TOP1_CURRENT_BAND`` of ``V_TOP1_TARGET``; ``fail`` outside it, ``skipped`` without an IgBLAST release (which
    leaves the audit incomplete);
  - ``band.sim_v_top1`` (informational): the same metric in the stored ``all`` run, consistent with the P2 record
    when within ``V_TOP1_BAND`` of ``V_TOP1_TARGET`` (ruling A1: read as recorded), ``warn`` otherwise: it was
    computed by older code, and the P2 number is where the target comes from;
  - ``band.speed`` (informational): IgBLAST's steady throughput, timed now on a fresh dev S1 by the speed protocol,
    within ``SPEED_BAND`` of the stored run's measurement; ``skipped (load)`` when the 1-minute load stays above
    ``SPEED_LOAD``, ``warn`` outside the band or on an error, never a failure.

**Read only.** The stored runs are only read: the test-split ``all`` run for its computed oracle and IgBLAST numbers
(it is never run or scored again), the ``real`` run for its records. Anything re-run goes into a fresh
``layout.audit_scratch(<check>)``, removed first, and the output directory may not overlap either run.
"""
from __future__ import annotations

from alignair_assay.workflow.audit import base, igblast, nulls, reproducible, run, stored
from alignair_assay.workflow.audit.base import (
    AUDIT_KIND,
    AUDIT_VERSION,
    STATUSES,
    GROUPS,
    CHECKS,
    INFORMATIONAL,
    GATING,
    VERDICTS,
    CHECKPOINT_LABEL,
    NULL_LABELS,
    REPRO_SUITE,
    REPRO_SCALE,
    REPRO_FILES,
    NULL_SUITES,
    DIAGNOSE_SUITES,
    NULL_SCALE,
    DIAGNOSE_BATCH,
    REAL_DONOR,
    REAL_CAP,
    POSITIVE_MODULE,
    POSITIVE_METRIC,
    AGREEMENT_TOOL,
    KNOWN_METRICS,
    KNOWN_FLOOR,
    V_TOP1_SUITE,
    V_TOP1_METRIC,
    V_TOP1_TARGET,
    V_TOP1_BAND,
    V_TOP1_CURRENT_BAND,
    V_TOP1_CURRENT_SCALE,
    IGBLAST,
    SPEED_BAND,
    SPEED_LOAD,
    SPEED_SETTLE_SECONDS,
    SPEED_POLL_SECONDS,
    SPEED_SCALE,
    SPEED_SUITE,
    CODE_FIELDS,
    SIM_HEADLINE,
    REAL_HEADLINE,
    HEAD_BOUNDARY_PREFIXES,
    AuditConfig,
    check,
    settle,
    verdict,
)
from alignair_assay.workflow.audit.stored import (
    sim_oracle,
    sim_staleness,
    dev_donors,
    real_summary,
    real_oracle,
    real_truth_health,
)
from alignair_assay.workflow.audit.igblast import (
    real_known_band,
    sim_v_top1_band,
    igblast_speed,
    igblast_v_top1_current,
    wait_for_idle,
)
from alignair_assay.workflow.audit.reproducible import (
    genairr_digest,
    genairr_identity,
    sim_reproducible,
    jsonl_content,
    compare_records,
    real_reproducible,
)
from alignair_assay.workflow.audit.nulls import (
    compare_problems,
    pairing,
    changes_problems,
    all_null,
    diagnose_null,
    ensure_built,
    shared_suite,
    positive_diagnose,
    checkpoint_model_key,
    all_vs_diagnose,
    real_null,
    real_comparison_problems,
)
from alignair_assay.workflow.audit.run import check_inputs, run_audit
