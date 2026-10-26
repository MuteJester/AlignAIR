"""Study summaries written from completed training diagnostics."""
from __future__ import annotations

import json
import statistics
from pathlib import Path

from alignair.aa.guard.loco import LOCO_MASK_HASH
from . import budget


def write_sweep_report(results, frozen, *, artifact_dir, manifest, device):
    artifact_dir = Path(artifact_dir)
    ok = [r for r in results.values() if r["status"] == "ok"]
    fail = [r for r in results.values() if r["status"] != "ok"]
    lines = ["# Stage 4 — Result\n",
             f"**{len(ok)}/{len(results)} runs OK** · manifest `{manifest['manifest_sha256'][:12]}` · "
             f"device {device} · budget {json.dumps(budget.total(path=artifact_dir / 'budget.json'))}\n",
             "Per-seed freezing uses the committed dev-only rule (min dev_total_loss, tie->lowest seed). "
             "Candidate eligibility and C1–C4 comparison are Stage 6, NOT decided here.\n",
             "## Frozen production seed per candidate\n",
             "| candidate | frozen seed | dev_loss | app_AUROC | type_acc | completeness_acc | energy_sep | eligible_src_frac |",
             "|---|---|---|---|---|---|---|---|"]
    for cid, seed in sorted(frozen.items()):
        r = results[f"{cid}.s{seed}"]
        lines.append(f"| {cid} | {seed} | {r['dev_total_loss']:.4f} | {r.get('applicability_auroc')} | "
                     f"{r.get('type_acc')} | {r.get('completeness_acc')} | {r.get('energy_separation')} | "
                     f"{r.get('eligible_source_fraction')} |")
    lines += ["\n## All runs\n",
              "| run | status | dev_loss | app_AUROC | pos_rec | neg_rec | type_acc | src_acc | e_in | e_oe | "
              "train/dev gap | stop | wall_s | vram_MB | params |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in sorted(results):
        r = results[tag]
        if r["status"] == "ok":
            lines.append(f"| {tag} | ok | {r['dev_total_loss']:.4f} | {r.get('applicability_auroc')} | "
                         f"{r.get('pos_recall')} | {r.get('neg_recall')} | {r.get('type_acc')} | "
                         f"{r.get('source_acc')} | {r.get('energy_in_mean')} | {r.get('energy_oe_mean')} | "
                         f"{r.get('train_dev_gap')} | {r.get('stop_step')} | {r.get('wall_s')} | "
                         f"{r.get('peak_vram_mb')} | {r.get('params')} |")
        else:
            lines.append(f"| {tag} | {r['status']} | — | — | — | — | — | — | — | — | — | {r.get('step')} | — | — | — |")
    if fail:
        lines.append("\n**Engineering failures preserved (reason recorded; only operational retries, no tuning):**")
        for r in fail:
            lines.append(f"- `{r['tag']}`: {r.get('reason')}")
    lines += ["\n## Scope guard", "- Fresh selection, blind-LOCO, far-OOD, and the sealed test were NOT loaded.",
              "- No candidate chosen; no threshold selected. Next: Stage 5 LOCO (stop for review)."]
    (artifact_dir / "STAGE4_RESULT.md").write_text("\n".join(lines) + "\n")


def write_loco_report(results, *, artifact_dir, manifest, device):
    artifact_dir = Path(artifact_dir)
    ok = [r for r in results.values() if r["status"] == "ok"]
    fail = [r for r in results.values() if r["status"] != "ok"]
    lines = ["# Stage 5 (a) — LOCO training result (burned-dev only; NO eligibility judgment)\n",
             f"**{len(ok)}/{len(results)} models OK** · loco manifest `{manifest['manifest_sha256'][:12]}` · "
             f"mask `{LOCO_MASK_HASH[:12]}` · device {device} · budget {json.dumps(budget.total(path=artifact_dir / 'budget.json'))}\n",
             "Candidates use frozen Stage-4 seeds + unchanged hyperparameters. Each fold restricts the frozen "
             "partition (no reshuffle); g excluded from train/dev/mining/norm/source-balance/early-stop; "
             "normalization + source vocab recomputed per fold. Blind features per fold. This is a training "
             "diagnostic — candidate eligibility and C1–C4 comparison remain Stage 6.\n",
             "## Per candidate × fold\n",
             "| model | fold | status | dev_loss | app_AUROC | type_acc | complete_acc | elig_src_frac | "
             "n_train | n_excl | stop | wall_s | vram_MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in sorted(results):
        r = results[tag]
        f = r["fold"]; ne = manifest["folds"][f]["n_excluded"]; nt = manifest["folds"][f]["n_train_after_exclusion"]
        if r["status"] == "ok":
            lines.append(f"| {tag} | {f} | ok | {r['dev_total_loss']:.4f} | {r.get('applicability_auroc')} | "
                         f"{r.get('type_acc')} | {r.get('completeness_acc')} | {r.get('eligible_source_fraction')} | "
                         f"{nt} | {ne} | {r.get('stop_step')} | {r.get('wall_s')} | {r.get('peak_vram_mb')} |")
        else:
            lines.append(f"| {tag} | {f} | {r['status']} | — | — | — | — | — | {nt} | {ne} | {r.get('step')} | — | — |")
    # per-candidate cross-fold stability (spread over the 4 folds)
    lines += ["\n## Cross-fold stability per candidate (dev_loss / AUROC spread over its 4 folds)\n",
              "| candidate | folds_ok | dev_loss mean±sd | AUROC mean±sd |", "|---|---|---|---|"]
    for cid in manifest["candidates"]:
        rs = [r for r in ok if r["cid"] == cid]
        if rs:
            dl = [r["dev_total_loss"] for r in rs]
            au = [r["applicability_auroc"] for r in rs if r.get("applicability_auroc") is not None]
            sd = (lambda x: statistics.pstdev(x) if len(x) > 1 else 0.0)
            lines.append(f"| {cid} | {len(rs)}/4 | {statistics.mean(dl):.4f}±{sd(dl):.4f} | "
                         f"{statistics.mean(au):.4f}±{sd(au):.4f} |" if au else
                         f"| {cid} | {len(rs)}/4 | {statistics.mean(dl):.4f}±{sd(dl):.4f} | unavailable |")
    if fail:
        lines.append("\n**Engineering failures preserved (reason recorded; operational retries only, no tuning):**")
        for r in fail:
            lines.append(f"- `{r['tag']}` (fold {r['fold']}): {r.get('reason')}")
    lines += ["\n## Scope audit", f"- files opened: {manifest['files_opened_audit']['opened']}",
              f"- NOT opened: {manifest['files_opened_audit']['NOT_opened']}",
              "- No eligibility judgment made. Next: Task 5.3 + fresh one-shot sequence (separate review)."]
    (artifact_dir / "STAGE5_LOCO_TRAINING.md").write_text("\n".join(lines) + "\n")
