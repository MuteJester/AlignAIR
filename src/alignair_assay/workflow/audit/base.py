"""The audit's checks as records: the check names, groups and verdicts, the configuration (``AuditConfig``) and its
sizes, ``check``, ``settle`` and ``verdict``, and the formatting the checks share."""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from alignair_assay.core.reference import DEFAULT_REFERENCE
from alignair_assay.runners.base import RunnerOptions
from alignair_assay.suites.engine import DEFAULT_SEED

AUDIT_KIND = "alignair_assay.audit"
# 2: the provenance names the code audit_code_sha256, taken at the start; 3: the gating checks and the incomplete
# verdict, the positive control and the cross-surface agreement, the GenAIRR freeze in the provenance
AUDIT_VERSION = "3"
STATUSES = ("pass", "fail", "warn", "skipped")
# The check groups, in report order, and every check in report order.
GROUPS = {"self_check": "Self-checks", "reproducible": "Reproducibility", "null": "Null comparisons",
          "positive": "Positive control", "agreement": "Cross-surface agreement", "band": "IgBLAST bands"}
CHECKS = ("self_check.sim_oracle", "self_check.sim_oracle_current", "self_check.sim_staleness",
          "self_check.real_oracle", "self_check.real_truth_health", "reproducible.sim_build",
          "reproducible.real_scores", "null.diagnose", "null.all", "null.real", "positive.diagnose",
          "agreement.all_vs_diagnose", "band.real_known", "band.sim_v_top1", "band.sim_v_top1_current", "band.speed")
# The informational checks: a warning at worst, never a failure, even when they raise.
INFORMATIONAL = ("self_check.sim_staleness", "band.sim_v_top1", "band.speed")
# The checks that can fail the audit (every other one), in report order: one of them skipped leaves it incomplete.
GATING = tuple(name for name in CHECKS if name not in INFORMATIONAL)
# The verdicts, and the command's exit status for each.
VERDICTS = {"pass": 0, "fail": 2, "incomplete": 3}
# The audited checkpoint's label (``--checkpoint base=PATH``) and the two labels it runs under in a null comparison.
CHECKPOINT_LABEL = "base"
NULL_LABELS = ("a", "b")
# Reproducibility: the suite built twice, its scale, and the files both builds must hold (donors.json only with donors).
REPRO_SUITE, REPRO_SCALE = "S1", 0.02
REPRO_FILES = ("cases", "truth", "reads")
# The null ``all`` comparison's suites (S1, and S7 for its genotype arm: the genotype path of ``all`` with checkpoint
# variants, about 150 reads at this scale), ``diagnose``'s, the scale of their one shared build, and the batch size
# ``diagnose`` replays with (the alignair-local runner's, so the two surfaces compute alike); ``real``'s donor and cap.
NULL_SUITES, DIAGNOSE_SUITES, NULL_SCALE = ("S1", "S7"), ("S1", "S3"), 0.05
DIAGNOSE_BATCH = RunnerOptions().batch_size
REAL_DONOR, REAL_CAP = "W-97", 2000
# The positive control: the module zeroed in a copy of the checkpoint, and the head metric that must get worse. The
# chip's parser places every boundary, so with it zeroed the J end under the logits' argmax, which the model gets
# right on most S1 reads, cannot stay exact.
POSITIVE_MODULE, POSITIVE_METRIC = "core.parser", "head.j.end_exact.argmax"
# The tool of ``all`` whose per-case scores must equal ``diagnose``'s (the checkpoint's second label).
AGREEMENT_TOOL = f"alignair-local@{NULL_LABELS[1]}"
# IgBLAST against independent evidence: its known answers on real reads, its S1 V top-1 (the P2 number) and speed.
KNOWN_METRICS = ("real.v.known_top1", "real.v.ref_known_top1")
KNOWN_FLOOR = 0.99
V_TOP1_SUITE, V_TOP1_METRIC = "S1", "v.top1_in_set"
V_TOP1_TARGET, V_TOP1_BAND = 0.979, 0.01
# IgBLAST's S1 V top-1 at this code (the gate): its band around the same target, and the scale of the dev S1 it is
# run on when the speed check built none.
V_TOP1_CURRENT_BAND, V_TOP1_CURRENT_SCALE = 0.015, 0.2
IGBLAST = "igblast"
SPEED_BAND = 0.15
# The 1-minute load above which the machine is too busy to time; how long to wait for it to settle, polled how often.
SPEED_LOAD = 1.5
SPEED_SETTLE_SECONDS = 180
SPEED_POLL_SECONDS = 5
# The scale of the fresh dev S1 the speed protocol times (as the stored run's: N_FULL of its reads, evenly spaced).
SPEED_SCALE = 1.0
SPEED_SUITE = "S1"
# A JSON-lines header's fields that name the code it was computed by: what may differ when only the code moved.
CODE_FIELDS = ("code_sha256", "fingerprint")
_EXAMPLES = 8
# The assay package (``alignair_assay``), whose sources the audit's code digest covers.
_PACKAGE = Path(__file__).resolve().parents[2]
_TIE = "tie_or_inconclusive"
# The end-to-end headline metrics a null comparison's n rests on: the fewest reads any of them pairs.
SIM_HEADLINE = ("clone_key.exact", "junction.nt_exact", "v.top1_in_set", "d.top1_in_set", "j.top1_in_set")
REAL_HEADLINE = ("real.v.unsupported", "real.v.known_top1", "real.v.ref_known_top1")
# The head family of a diagnose comparison holds a boundary metric of each gene and edge; one pairs only the reads
# both models orient as the truth, so it must pair some, not all.
HEAD_BOUNDARY_PREFIXES = tuple(f"head.{gene}.{edge}_" for gene in "vdj" for edge in ("start", "end"))


@dataclass(frozen=True)
class AuditConfig:
    """One audit: the stored ``all`` run (``sim_run``) and ``real`` run, the data root of the real reads, the audited
    ``checkpoint`` (a ``.alignair`` file) and the torch ``device`` the null comparisons run it on, whether to time
    IgBLAST (``skip_speed``), the IgBLAST release to time (None: the one the ``all`` run recorded), the build
    ``workers`` and tool ``threads``, the suites' ``seed`` and ``reference``; then the ruling's sizes, which only a
    test shrinks: the reproducibility build, the null comparisons' suites (``all``'s and ``diagnose``'s), the scale
    of their shared build, ``diagnose``'s batch size, the donor and cap, the diagnosed ``diagnose_variants`` (None:
    every one), the speed protocol's S1 scale and the scale of the S1 IgBLAST's current V top-1 is measured on when
    the speed check built none."""

    sim_run: str
    real_run: str
    data_root: str
    checkpoint: str
    device: str = "cpu"
    skip_speed: bool = False
    igblast_dir: str | None = None
    workers: int = 8
    threads: int = 8
    seed: int = DEFAULT_SEED
    reference: str = DEFAULT_REFERENCE
    repro_suite: str = REPRO_SUITE
    repro_scale: float = REPRO_SCALE
    null_suites: tuple[str, ...] = NULL_SUITES
    null_scale: float = NULL_SCALE
    diagnose_suites: tuple[str, ...] = DIAGNOSE_SUITES
    batch_size: int = DIAGNOSE_BATCH
    diagnose_variants: tuple[str, ...] | None = None
    real_donor: str = REAL_DONOR
    real_cap: int = REAL_CAP
    speed_scale: float = SPEED_SCALE
    v_top1_scale: float = V_TOP1_CURRENT_SCALE


# --- checks -------------------------------------------------------------------------------------------------------

def check(name: str, status: str, n: int, evidence: dict, reason: str | None = None) -> dict:
    """A check record: ``name`` (of ``CHECKS``), ``status`` (of ``STATUSES``), ``n`` (the reads or cases it rests
    on), ``evidence`` (with a one-line ``summary``) and the ``reason`` for a status other than pass."""
    if status not in STATUSES:
        raise ValueError(f"unknown check status {status!r}")
    return {"name": name, "status": status, "n": int(n), "evidence": evidence, "reason": reason}


def settle(found: dict) -> dict:
    """``found``, failed when it passed on nothing (``n`` below 1): a check that checked nothing never passes."""
    if found["status"] == "pass" and found["n"] < 1:
        return {**found, "status": "fail", "reason": "checked nothing (n = 0)"}
    return found


def verdict(checks: list[dict]) -> dict:
    """The audit's verdict (of ``VERDICTS``): ``fail`` when a check failed; else ``incomplete`` when a check that can
    fail the audit (``GATING``) was skipped, since what was not checked may have failed; else ``fail`` when no check
    passed (an audit that checked nothing never passes); else ``pass``. With the ``counts`` per status, the
    ``failed`` and ``skipped_gating`` check names and the one-line ``line`` that opens ``audit.md``."""
    counts = Counter(found["status"] for found in checks)
    failed = [found["name"] for found in checks if found["status"] == "fail"]
    skipped = [found["name"] for found in checks if found["status"] == "skipped" and found["name"] in GATING]
    checked = any(found["status"] == "pass" for found in checks)
    status = "fail" if failed else "incomplete" if skipped else "pass" if checked else "fail"
    line = (f"AUDIT {status.upper()}: {len(checks)} checks, " +
            ", ".join(f"{counts[name]} {name}" for name in STATUSES))
    if failed:
        line += f"; failed: {', '.join(failed)}"
    if skipped:
        line += f"; skipped, and able to fail the audit: {', '.join(skipped)}"
    if not checked:
        line += "; nothing was checked"
    return {"status": status, "counts": {name: counts[name] for name in STATUSES}, "failed": failed,
            "skipped_gating": skipped, "line": line}


def _signed(value) -> str:
    return "—" if value is None else "0" if value == 0 else f"{value:+.6g}"


def _percent(value, digits: int = 1) -> str:
    return "—" if value is None else f"{100 * value:.{digits}f}%"


def _rate(value) -> str:
    return "—" if not value else f"{value:.1f} reads/s"
