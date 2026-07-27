"""Stage-4 seed sweep — train C1-C4 x 3 seeds on the burned Gate-3 dev TRAIN partition, freeze one seed per
candidate by the committed dev-only rule, record the budget, and write STAGE4_RESULT.md. Does NOT choose the
best candidate (that is Stage 6). Fresh selection, blind-LOCO, far-OOD, and the seal are never loaded.

Resumable: each run's checkpoint holds model+optimizer+scheduler+scaler+RNG state. OOM / NaN / data-contract
failure / representation collapse are treated as ENGINEERING failures (artifact + reason preserved; only
operational failures are retried from the same config) — never a reason to change scientific thresholds.

Run:  PYTHONPATH=src nohup .venv/bin/python scripts/aa/applicability/run_stage4_sweep.py > <log> 2>&1 &
"""
from __future__ import annotations

import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path("scripts/aa/applicability").resolve()))
import budget  # noqa: E402

from alignair.aa.guard.checkpoint import build_provenance  # noqa: E402
from alignair.aa.guard.classes import COMPLETENESS_INDEX, DOMAIN_TYPE_INDEX  # noqa: E402
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features  # noqa: E402
from alignair.aa.guard.fusion import fit_fusion  # noqa: E402
from alignair.aa.guard.losses import energy_margin_loss, supcon_loss  # noqa: E402
from alignair.aa.guard.model import GuardModel  # noqa: E402
from alignair.aa.guard.profiles import load_bank  # noqa: E402
from alignair.aa.guard.tokenizer import MAX_LEN, AaTokenizer, enforce_length_contract  # noqa: E402
from alignair.aa.guard.train_data import partition_by_parent_lineage  # noqa: E402

ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
MAN = json.load(open(ART / "stage4_run_manifest.json"))
DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")
PROF = Path(".private/aa-applicability/profiles")
CKPT = Path(MAN["checkpoint_dir"]); CKPT.mkdir(parents=True, exist_ok=True)
FEAT_CACHE = Path(".private/aa-applicability/dev_hmm_features.json")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

COMPLETE_MAP = {"cdr3_only": "too_short", "j_only": "missing_v_anchor", "v_only": "missing_j_anchor",
                "leader_vh": "contaminated", "vh_constant": "contaminated"}   # reversed_vh -> masked (no class)


def _domain_type(rec):
    g, cat = rec.get("group"), rec.get("category", "")
    if g in ("positive_real", "positive_sim"):
        return DOMAIN_TYPE_INDEX["human_igh_vh"]
    if g == "vhh_nonhuman":
        return DOMAIN_TYPE_INDEX["murine_vh"] if cat in ("murine_vh", "chimeric_murine_variable") \
            else DOMAIN_TYPE_INDEX["vhh"]
    if g == "igsf":
        return DOMAIN_TYPE_INDEX["igsf"]
    if g == "igk_igl":
        return DOMAIN_TYPE_INDEX["human_igk"] if cat == "igk" else DOMAIN_TYPE_INDEX["human_igl"]
    if g == "tcr":
        return DOMAIN_TYPE_INDEX["tra"] if cat == "tra" else DOMAIN_TYPE_INDEX["trb"]
    return -1                                            # structural / other -> no domain-type label


def _completeness(rec):
    if rec.get("group") in ("positive_real", "positive_sim"):
        return COMPLETENESS_INDEX["complete"]
    c = COMPLETE_MAP.get(rec.get("category", ""))
    return COMPLETENESS_INDEX[c] if c else -1            # reversed_vh / others -> masked


def precompute_features(recs, bank):
    cache = json.loads(FEAT_CACHE.read_text()) if FEAT_CACHE.exists() else {}
    import hashlib
    changed = False
    for r in recs:
        h = hashlib.sha256(r["aa"].encode()).hexdigest()
        if h not in cache:
            cache[h] = hmm_features(bank, r["aa"]).to_vector().tolist()
            changed = True
    if changed:
        FEAT_CACHE.write_text(json.dumps(cache))
    return cache


def make_oe(train, seed, cap_n):
    rng = random.Random(seed)
    oe = []
    for r in rng.sample(train, min(len(train), cap_n * 4)):
        chars = list(r["aa"])
        rng.shuffle(chars)                               # residue-shuffle destroys the domain -> outlier
        oe.append({"id": f"oe::{r['id']}", "aa": "".join(chars), "group": "oe"})
    return oe


def _source_setup(train):
    """Category as a source proxy; a domain-type is source-adv ELIGIBLE only if it has >=2 source categories
    (Safeguard 4). Returns (cat2id, eligible_domain_types)."""
    from collections import defaultdict
    cats_by_dt = defaultdict(set)
    for r in train:
        dt = _domain_type(r)
        if dt >= 0:
            cats_by_dt[dt].add(r.get("category", ""))
    all_cats = sorted({c for cats in cats_by_dt.values() for c in cats})
    cat2id = {c: i for i, c in enumerate(all_cats)}
    eligible = {dt for dt, cats in cats_by_dt.items() if len(cats) >= 2}
    return cat2id, eligible, len(all_cats)


def build_tensors(records, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt, is_oe=False):
    import hashlib
    ids_l, mask_l, hmm_l = [], [], []
    dt_t, dt_m, cp_t, cp_m, app_t, en_out, en_m, src_t, src_m, ct_t, ct_m = ([] for _ in range(11))
    for r in records:
        if not enforce_length_contract(r["aa"]).ok:
            continue
        h = hashlib.sha256(r["aa"].encode()).hexdigest()
        vec = feat_cache.get(h) or hmm_features(load_bank(PROF), r["aa"]).to_vector().tolist()
        hmm_l.append(fusion.normalize(vec, feature_names=FEATURE_NAMES, bank_checksum=bank_checksum))
        i, m = tok.encode(r["aa"]); ids_l.append(i); mask_l.append(m)
        if is_oe:
            dt_t.append(0); dt_m.append(0.0); cp_t.append(0); cp_m.append(0.0)
            app_t.append(0.0); en_out.append(1.0); en_m.append(1.0)
            src_t.append(0); src_m.append(0.0); ct_t.append(0); ct_m.append(0.0)
            continue
        dt = _domain_type(r); cp = _completeness(r)
        dt_t.append(max(dt, 0)); dt_m.append(1.0 if dt >= 0 else 0.0)
        cp_t.append(max(cp, 0)); cp_m.append(1.0 if cp >= 0 else 0.0)
        app_t.append(1.0 if r.get("expected_state") == "supported" else 0.0)
        en_out.append(0.0); en_m.append(1.0 if dt >= 0 else 0.0)      # in-domain samples only
        elig = dt >= 0 and dt in eligible_dt
        src_t.append(cat2id.get(r.get("category", ""), 0)); src_m.append(1.0 if elig else 0.0)
        ct_t.append(max(dt, 0)); ct_m.append(1.0 if dt >= 0 else 0.0)
    t = lambda x, dt_: torch.tensor(x, dtype=dt_)  # noqa: E731
    return {"ids": torch.stack(ids_l), "mask": torch.stack(mask_l), "hmm": torch.stack(hmm_l),
            "domain_type": t(dt_t, torch.long), "type_m": t(dt_m, torch.float),
            "completeness": t(cp_t, torch.long), "comp_m": t(cp_m, torch.float),
            "applicability": t(app_t, torch.float),
            "energy_out": t(en_out, torch.float), "energy_m": t(en_m, torch.float),
            "source": t(src_t, torch.long), "source_m": t(src_m, torch.float),
            "contrast": t(ct_t, torch.long), "contrast_m": t(ct_m, torch.float)}


def _forward_losses(model, T, W, margins, cfg, grl):
    out = model(T["ids"].to(DEVICE), T["mask"].to(DEVICE),
                hmm_norm=T["hmm"].to(DEVICE) if cfg["fusion"] else None, grl_lambda=grl)
    dev = DEVICE
    parts = {}
    tm = T["type_m"].to(dev)
    parts["type"] = (F.cross_entropy(out["domain_type"], T["domain_type"].to(dev), reduction="none") * tm
                     ).sum() / tm.sum().clamp(min=1)
    cm = T["comp_m"].to(dev)
    parts["complete"] = (F.cross_entropy(out["completeness"], T["completeness"].to(dev), reduction="none") * cm
                         ).sum() / cm.sum().clamp(min=1)
    parts["app"] = F.binary_cross_entropy_with_logits(out["applicability"].squeeze(-1), T["applicability"].to(dev))
    parts["energy"] = energy_margin_loss(out["energy"], T["energy_out"].to(dev), T["energy_m"].to(dev),
                                         m_in=margins["m_in"], m_out=margins["m_out"]) if cfg["energy"] \
        else out["energy"].sum() * 0.0
    parts["contrast"] = supcon_loss(out["rep"], T["contrast"].to(dev), T["contrast_m"].to(dev)) if cfg["contrast"] \
        else out["rep"].sum() * 0.0
    if cfg["source"]:
        sm = T["source_m"].to(dev)
        parts["source"] = (F.cross_entropy(out["source_logits"], T["source"].to(dev), reduction="none") * sm
                           ).sum() / sm.sum().clamp(min=1) if sm.sum() > 0 else out["source_logits"].sum() * 0.0
    else:
        parts["source"] = out["source_logits"].sum() * 0.0
    total = (parts["type"] + W["complete"] * parts["complete"] + W["app"] * parts["app"]
             + W["oe"] * parts["energy"] + W["contrast"] * parts["contrast"] + W["source"] * parts["source"])
    return total, parts, out


def _rng_state():
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if DEVICE == "cuda" else None,
            "python": random.getstate(), "numpy": np.random.get_state()}


def train_one(cid, seed, train, dev, oe, feat_cache, cat2id, eligible_dt, n_sources):
    tag = f"{cid}.s{seed}"
    ckpt_path = CKPT / f"{tag}.pt"
    cfg = GuardModel(cid, hmm_dim=len(FEATURE_NAMES)).cfg
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr = MAN["training"]; W = MAN["loss_weights"]; margins = MAN["energy_margins"]
    bank_checksum = MAN["bank_checksum"]

    fusion = fit_fusion(FEATURE_NAMES, [feat_cache[__import__("hashlib").sha256(r["aa"].encode()).hexdigest()]
                                        for r in train], bank_checksum)   # TRAIN-only normalization
    tok = AaTokenizer(max_len=MAX_LEN)
    T_dev = build_tensors(dev, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt)
    T_oe = build_tensors(oe, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt, is_oe=True) if oe else None

    model = GuardModel(cid, hmm_dim=len(FEATURE_NAMES)).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=MAN["optimizer"]["lr"],
                            weight_decay=MAN["optimizer"]["weight_decay"])
    max_steps, warm = tr["max_steps"], MAN["schedule"]["warmup_steps"]
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))
    scaler = torch.cuda.amp.GradScaler(enabled=(DEVICE == "cuda" and tr["amp"]))
    start_step, best_dev, patience, best_state = 0, float("inf"), 0, None

    if ckpt_path.exists():                               # resume operationally
        st = torch.load(ckpt_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); start_step = st["step"]; best_dev = st["best_dev"]
        patience = st["patience"]; best_state = st.get("best_state")
        random.setstate(st["rng"]["python"]); np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"])
        if DEVICE == "cuda" and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all(st["rng"]["cuda"])

    if DEVICE == "cuda":
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed + 1000)
    by_role = {"positive": [r for r in train if r["group"] in ("positive_real", "positive_sim")],
               "biological_negative": [r for r in train if r["group"] in ("vhh_nonhuman", "igsf", "igk_igl", "tcr")],
               "structural": [r for r in train if r["group"] == "structural"]}
    roles = [k for k in by_role if by_role[k]]
    bs = tr["batch_size"]; n_oe = int(bs * MAN["oe"]["cap_fraction"]) if oe else 0
    per_role = max(1, (bs - n_oe) // len(roles))
    t0 = time.time(); stop_step = start_step; train_loss_ema = None
    try:
        model.train()
        for step in range(start_step, max_steps):
            batch = [rng.choice(by_role[r]) for r in roles for _ in range(per_role)]
            Tb = build_tensors(batch, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt)
            if oe and n_oe:
                Toe = build_tensors([rng.choice(oe) for _ in range(n_oe)], feat_cache, fusion, tok,
                                    bank_checksum, cat2id, eligible_dt, is_oe=True)
                for k in Tb:
                    Tb[k] = torch.cat([Tb[k], Toe[k]], dim=0)
            opt.zero_grad()
            with torch.cuda.amp.autocast(enabled=(DEVICE == "cuda" and tr["amp"])):
                total, parts, _ = _forward_losses(model, Tb, W, margins, cfg, MAN["grl_lambda"])
            if not torch.isfinite(total):
                raise FloatingPointError(f"non-finite loss at step {step}")
            scaler.scale(total).backward()
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"])
            scaler.step(opt); scaler.update(); sched.step()
            train_loss_ema = float(total) if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * float(total)

            if (step + 1) % tr["eval_every"] == 0:
                model.eval()
                with torch.no_grad():
                    dtot, dparts, dout = _forward_losses(model, T_dev, W, margins, cfg, MAN["grl_lambda"])
                    if T_oe is not None:
                        _, _, ooe = _forward_losses(model, T_oe, W, margins, cfg, MAN["grl_lambda"])
                dv = float(dtot)
                if dv < best_dev - tr["min_delta"]:
                    best_dev, patience, stop_step = dv, 0, step + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step + 1, "best_dev": best_dev,
                            "patience": patience, "best_state": best_state, "rng": _rng_state()}, ckpt_path)
                model.train()
                if patience >= tr["early_stop_patience"]:
                    break
    except (torch.cuda.OutOfMemoryError, FloatingPointError, RuntimeError) as e:  # engineering failure
        return {"tag": tag, "status": "engineering_failure", "reason": f"{type(e).__name__}: {str(e)[:120]}",
                "step": stop_step}

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _evaluate(model, T_dev, T_oe, cfg, W, margins)
    metrics.update({"tag": tag, "cid": cid, "seed": seed, "status": "ok", "stop_step": stop_step,
                    "dev_total_loss": best_dev, "train_loss_ema": round(train_loss_ema or 0.0, 4),
                    "train_dev_gap": round((train_loss_ema or 0.0) - best_dev, 4),
                    "wall_s": round(time.time() - t0, 1),
                    "gpu_h": round((time.time() - t0) / 3600, 4),
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if DEVICE == "cuda" else 0,
                    "params": sum(p.numel() for p in model.parameters()),
                    "eligible_source_fraction": round(float(T_dev["source_m"].mean()), 4)})
    prov = build_provenance(bank_checksum=bank_checksum, training_manifest_hash=MAN["manifest_sha256"],
                            partition_hash=MAN["train_partition_hash"], seed=seed,
                            loss_mask_config={"candidate": cid, "weights": W, "margins": margins})
    torch.save({"model": model.state_dict(), "provenance": prov.to_dict(), "metrics": metrics},
               CKPT / f"{tag}.final.pt")
    budget.record(tag, metrics["wall_s"], metrics["gpu_h"], metrics["params"])
    return metrics


def _evaluate(model, T_dev, T_oe, cfg, W, margins):
    model.eval()
    with torch.no_grad():
        _, parts, out = _forward_losses(model, T_dev, W, margins, cfg, 1.0)
        app_logit = out["applicability"].squeeze(-1).cpu().numpy()
        app_y = T_dev["applicability"].numpy()
        dt_pred = out["domain_type"].argmax(-1).cpu().numpy(); dt_y = T_dev["domain_type"].numpy()
        dt_m = T_dev["type_m"].numpy().astype(bool)
        cp_pred = out["completeness"].argmax(-1).cpu().numpy(); cp_y = T_dev["completeness"].numpy()
        cp_m = T_dev["comp_m"].numpy().astype(bool)
        src_pred = out["source_logits"].argmax(-1).cpu().numpy(); src_y = T_dev["source"].numpy()
        src_m = T_dev["source_m"].numpy().astype(bool)
        e_in = out["energy"].cpu().numpy()
        e_oe = None
        if T_oe is not None:
            _, _, ooe = _forward_losses(model, T_oe, W, margins, cfg, 1.0)
            e_oe = ooe["energy"].cpu().numpy()

    def _acc(pred, y, m):
        return round(float((pred[m] == y[m]).mean()), 4) if m.any() else None

    def _auroc(y, s):
        try:
            return round(float(roc_auc_score(y, s)), 4)
        except Exception:
            return None
    pos, neg = app_y == 1, app_y == 0
    return {
        "component_losses": {k: round(float(v), 4) for k, v in parts.items()},
        "type_acc": _acc(dt_pred, dt_y, dt_m), "completeness_acc": _acc(cp_pred, cp_y, cp_m),
        "applicability_auroc": _auroc(app_y, app_logit),
        "pos_recall": round(float(((app_logit > 0) & pos).sum() / max(1, pos.sum())), 4),
        "neg_recall": round(float(((app_logit <= 0) & neg).sum() / max(1, neg.sum())), 4),
        "source_acc": _acc(src_pred, src_y, src_m),
        "energy_in_mean": round(float(e_in.mean()), 3),
        "energy_oe_mean": round(float(e_oe.mean()), 3) if e_oe is not None else None,
        "energy_separation": round(float(e_oe.mean() - e_in.mean()), 3) if e_oe is not None else None,
    }


def main():
    recs = [json.loads(l) for l in open(DEV)]
    train, dev, part_hash = partition_by_parent_lineage(recs, seed=42, dev_frac=0.2)
    assert part_hash == MAN["train_partition_hash"], "partition drift vs frozen manifest"
    bank = load_bank(PROF)
    feat_cache = precompute_features(recs, bank)
    oe = make_oe(train, seed=42, cap_n=int(MAN["training"]["batch_size"] * MAN["oe"]["cap_fraction"]))
    cat2id, eligible_dt, n_sources = _source_setup(train)
    print(f"[sweep] device={DEVICE} n_train={len(train)} n_dev={len(dev)} n_oe={len(oe)} n_sources={n_sources}",
          flush=True)

    results = {}
    for cid in MAN["candidates"]:
        for seed in MAN["seeds"]:
            r = train_one(cid, seed, train, dev, oe, feat_cache, cat2id, eligible_dt, n_sources)
            results[r["tag"]] = r
            print(f"[sweep] {r['tag']} status={r['status']} "
                  f"dev_loss={r.get('dev_total_loss')} app_auroc={r.get('applicability_auroc')} "
                  f"stop={r.get('stop_step')} wall={r.get('wall_s')}s", flush=True)
            (ART / "stage4_runs.json").write_text(json.dumps(results, indent=2, sort_keys=True))

    # freeze one seed per candidate by the committed dev-only rule (NOT choosing the best candidate)
    from train_candidate import choose_production_seed
    frozen = {}
    for cid in MAN["candidates"]:
        dev_results = {r["seed"]: {"dev_loss": r["dev_total_loss"]} for r in results.values()
                       if r["cid"] == cid and r["status"] == "ok"}
        if dev_results:
            frozen[cid] = choose_production_seed(dev_results)

    _write_result(results, frozen)
    print("[sweep] DONE — STAGE4_RESULT.md written; stopping for review before LOCO.", flush=True)


def _write_result(results, frozen):
    ok = [r for r in results.values() if r["status"] == "ok"]
    fail = [r for r in results.values() if r["status"] != "ok"]
    lines = ["# Stage 4 — Result\n",
             f"**{len(ok)}/{len(results)} runs OK** · manifest `{MAN['manifest_sha256'][:12]}` · "
             f"device {DEVICE} · budget {json.dumps(budget.total())}\n",
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
    (ART / "STAGE4_RESULT.md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
