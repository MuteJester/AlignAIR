"""Stage-5 step (a) — train the 16 LOCO models (4 candidates x 4 folds) on burned development data only.

Per the frozen loco_run_manifest.json: candidates use their FROZEN Stage-4 seeds + unchanged hyperparameters;
the frozen Stage-4 parent-lineage partition is RESTRICTED (fold group g removed; remaining records keep their
assignment — no reshuffle); g is excluded from train/dev/mining/normalization/source-balancing/early-stop;
HMM normalization is fit per fold on the remaining train; source-adversarial eligibility + the ordered source
vocabulary are recomputed per fold; each checkpoint binds LOCO_MASK_HASH, bank checksum, class-map hash,
partition hash, normalization-stat hash, and the excluded-record digest. Blind features per fold (the fold's
profiles masked). Resumable; Stage-4 engineering-retry policy. NO eligibility judgment. Stops before Task 5.3.

Fresh selection, blind-LOCO challenge, far-OOD, and the sealed test are never opened.
"""
from __future__ import annotations

import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path("scripts/aa/applicability").resolve()))
import budget  # noqa: E402
import run_stage4_sweep as S  # noqa: E402

from alignair.aa.guard.loco import LOCO_FOLDS, LOCO_MASK_HASH, masked_profiles_for  # noqa: E402
from alignair.aa.guard.profiles import load_bank  # noqa: E402
from alignair.aa.guard.train_data import partition_by_parent_lineage  # noqa: E402

ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
LOCO_MAN = json.load(open(ART / "loco_run_manifest.json"))
DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")
LOCO_CKPT = Path(LOCO_MAN["checkpoint_dir"])
BLIND_DIR = Path(".private/aa-applicability/loco_features"); BLIND_DIR.mkdir(parents=True, exist_ok=True)


def main():
    recs = [json.loads(l) for l in open(DEV)]
    train, dev, ph = partition_by_parent_lineage(recs, seed=42, dev_frac=0.2)
    assert ph == LOCO_MAN["base_partition_hash"], "partition drift vs LOCO manifest"
    frozen = LOCO_MAN["frozen_seeds"]
    bank = load_bank(S.PROF)
    print(f"[loco] device={S.DEVICE} folds={list(LOCO_FOLDS)} frozen_seeds={frozen}", flush=True)

    results = {}
    for g in LOCO_FOLDS:
        masked = masked_profiles_for(g)
        loco_train = [r for r in train if r.get("group") != g]        # restrict (no reshuffle)
        loco_dev = [r for r in dev if r.get("group") != g]
        cache_path = BLIND_DIR / f"{g}.json"
        oe = S.make_oe(loco_train, seed=42, cap_n=int(S.MAN["training"]["batch_size"] * S.MAN["oe"]["cap_fraction"]))
        blind = S.precompute_features(loco_train + loco_dev + oe, bank, mask_groups=masked, cache_path=cache_path)
        cat2id, eligible, nsrc = S._source_setup(loco_train)          # recompute from fold train
        assert sorted(cat2id) == LOCO_MAN["folds"][g]["source_vocab_ordered"], f"source vocab drift for {g}"
        assert nsrc == LOCO_MAN["folds"][g]["n_sources"]
        xp_base = {"loco_mask_hash": LOCO_MASK_HASH, "fold": g,
                   "masked_profiles": sorted(masked),
                   "partition_hash": LOCO_MAN["folds"][g]["loco_train_ids_sha256"],
                   "excluded_seqhash_sha256": LOCO_MAN["folds"][g]["excluded_seqhash_sha256"],
                   "training_manifest_hash": LOCO_MAN["manifest_sha256"]}
        for cid in LOCO_MAN["candidates"]:
            seed = frozen[cid]
            run_tag = f"{cid}.s{seed}.{g}"
            t0 = time.time()
            r = S.train_one(cid, seed, loco_train, loco_dev, oe, blind, cat2id, eligible, nsrc,
                            mask_groups=masked, ckpt_dir=LOCO_CKPT, extra_provenance=dict(xp_base), run_tag=run_tag)
            r["fold"] = g; r["cid"] = cid; r.setdefault("wall_s", round(time.time() - t0, 1))
            results[run_tag] = r
            print(f"[loco] {run_tag} status={r['status']} dev_loss={r.get('dev_total_loss')} "
                  f"auroc={r.get('applicability_auroc')} elig_src={r.get('eligible_source_fraction')} "
                  f"stop={r.get('stop_step')} wall={r.get('wall_s')}s", flush=True)
            (ART / "loco_runs.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    _write_result(results)
    print("[loco] DONE — 16 LOCO models trained; STOP before Task 5.3 / any fresh-data access.", flush=True)


def _write_result(results):
    ok = [r for r in results.values() if r["status"] == "ok"]
    fail = [r for r in results.values() if r["status"] != "ok"]
    lines = ["# Stage 5 (a) — LOCO training result (burned-dev only; NO eligibility judgment)\n",
             f"**{len(ok)}/{len(results)} models OK** · loco manifest `{LOCO_MAN['manifest_sha256'][:12]}` · "
             f"mask `{LOCO_MASK_HASH[:12]}` · device {S.DEVICE} · budget {json.dumps(budget.total())}\n",
             "Candidates use frozen Stage-4 seeds + unchanged hyperparameters. Each fold restricts the frozen "
             "partition (no reshuffle); g excluded from train/dev/mining/norm/source-balance/early-stop; "
             "normalization + source vocab recomputed per fold. Blind features per fold. This is a training "
             "diagnostic — candidate eligibility and C1–C4 comparison remain Stage 6.\n",
             "## Per candidate × fold\n",
             "| model | fold | status | dev_loss | app_AUROC | type_acc | complete_acc | elig_src_frac | "
             "n_train | n_excl | stop | wall_s | vram_MB |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for tag in sorted(results):
        r = results[tag]
        f = r["fold"]; ne = LOCO_MAN["folds"][f]["n_excluded"]; nt = LOCO_MAN["folds"][f]["n_train_after_exclusion"]
        if r["status"] == "ok":
            lines.append(f"| {tag} | {f} | ok | {r['dev_total_loss']:.4f} | {r.get('applicability_auroc')} | "
                         f"{r.get('type_acc')} | {r.get('completeness_acc')} | {r.get('eligible_source_fraction')} | "
                         f"{nt} | {ne} | {r.get('stop_step')} | {r.get('wall_s')} | {r.get('peak_vram_mb')} |")
        else:
            lines.append(f"| {tag} | {f} | {r['status']} | — | — | — | — | — | {nt} | {ne} | {r.get('step')} | — | — |")
    # per-candidate cross-fold stability (spread over the 4 folds)
    lines += ["\n## Cross-fold stability per candidate (dev_loss / AUROC spread over its 4 folds)\n",
              "| candidate | folds_ok | dev_loss mean±sd | AUROC mean±sd |", "|---|---|---|---|"]
    for cid in LOCO_MAN["candidates"]:
        rs = [r for r in ok if r["cid"] == cid]
        if rs:
            dl = [r["dev_total_loss"] for r in rs]
            au = [r["applicability_auroc"] for r in rs if r.get("applicability_auroc") is not None]
            sd = (lambda x: statistics.pstdev(x) if len(x) > 1 else 0.0)
            lines.append(f"| {cid} | {len(rs)}/4 | {statistics.mean(dl):.4f}±{sd(dl):.4f} | "
                         f"{statistics.mean(au):.4f}±{sd(au):.4f} |")
    if fail:
        lines.append("\n**Engineering failures preserved (reason recorded; operational retries only, no tuning):**")
        for r in fail:
            lines.append(f"- `{r['tag']}` (fold {r['fold']}): {r.get('reason')}")
    lines += ["\n## Scope audit", f"- files opened: {LOCO_MAN['files_opened_audit']['opened']}",
              f"- NOT opened: {LOCO_MAN['files_opened_audit']['NOT_opened']}",
              "- No eligibility judgment made. Next: Task 5.3 + fresh one-shot sequence (separate review)."]
    (ART / "STAGE5_LOCO_TRAINING.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
