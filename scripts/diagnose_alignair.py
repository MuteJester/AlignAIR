"""Full post-training diagnosis of a trained AlignAIR checkpoint via ModelXRay.

Loads a checkpoint (old pre-refactor keys are auto-remapped to the unified AlignAIR), builds a fresh
held-out probe from the gym, and runs BOTH X-ray tiers on the final model:

  * live tier   (ModelXRay.observe): grad/weight/update health, activation health, feature-space
                geometry, multi-task gradient conflict, Kendall weights, per-task eval.
  * heavy tier  (ModelXRay.deep_report): weightwatcher power-law alpha (heavy-tail self-regularization
                / generalization), Hessian top-eigenvalue + trace (loss-surface sharpness / flatness),
                CKA (inter-layer redundancy), neural collapse NC1-3 + classification margins on the V head.

Usage:  PYTHONPATH=src python scripts/diagnose_alignair.py --ckpt .private/models/alignair_igh.pt
"""
from __future__ import annotations

import argparse
import itertools
import json
import re

import torch

import GenAIRR.data as gd
from alignair.config.alignair_config import AlignAIRConfig
from alignair.gym import Curriculum
from alignair.models import AlignAIR
from alignair.models.losses import hierarchical_loss, make_logvars
from alignair.reference.reference_set import ReferenceSet
from alignair.training.alignair_trainer import _stream_records, build_batch, eval_metrics
from alignair.xray import ModelXRay


def remap_legacy_state_dict(old: dict) -> dict:
    """Pre-refactor SingleChain/MultiChain state_dict keys -> unified AlignAIR keys (no-op if already new)."""
    new = {}
    for k, v in old.items():
        nk = k
        if (m := re.match(r"seg_towers\.([vdj])\.(.*)", k)):             nk = f"branches.{m[1]}.seg_tower.{m[2]}"
        elif (m := re.match(r"seg_heads\.([vdj])_(start|end)\.(.*)", k)): nk = f"branches.{m[1]}.{m[2]}_head.{m[3]}"
        elif (m := re.match(r"cls_towers\.([vdj])\.(.*)", k)):           nk = f"branches.{m[1]}.cls_tower.{m[2]}"
        elif (m := re.match(r"cls_mid\.([vdj])\.(.*)", k)):             nk = f"branches.{m[1]}.cls_mid.{m[2]}"
        elif (m := re.match(r"cls_head\.([vdj])\.(.*)", k)):            nk = f"branches.{m[1]}.cls_head.{m[2]}"
        elif (m := re.match(r"mutation_rate_(mid|head)\.(.*)", k)):     nk = f"meta_heads.mutation_rate.{m[1]}.{m[2]}"
        elif (m := re.match(r"indel_count_(mid|head)\.(.*)", k)):       nk = f"meta_heads.indel_count.{m[1]}.{m[2]}"
        elif (m := re.match(r"productive_head\.(.*)", k)):              nk = f"meta_heads.productive.head.{m[1]}"
        elif (m := re.match(r"chain_type_(mid|head)\.(.*)", k)):        nk = f"meta_heads.chain_type_logits.{m[1]}.{m[2]}"
        new[nk] = v
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=".private/models/alignair_igh.pt")
    ap.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--probe-size", type=int, default=256)
    ap.add_argument("--progress", type=float, default=0.9, help="probe difficulty (gym curriculum)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--out", default=None, help="JSON diagnosis output (default: <ckpt>.diagnosis.json)")
    a = ap.parse_args()
    dev = a.device

    ck = torch.load(a.ckpt, map_location=dev, weights_only=False)
    cfg = AlignAIRConfig(**ck["config"])
    model = AlignAIR(cfg).to(dev)
    model.load_state_dict(remap_legacy_state_dict(ck["model"]), strict=True)
    logvars = make_logvars(cfg)
    try:
        logvars.load_state_dict(ck["logvars"])
    except Exception:
        pass
    logvars.to(dev)
    print(f"loaded {a.ckpt} @ step {ck.get('step')}  |  V={cfg.v_allele_count} D={cfg.d_allele_count} "
          f"J={cfg.j_allele_count}  params={sum(p.numel() for p in model.parameters())/1e6:.2f}M", flush=True)

    # ---- fresh held-out probe from the gym ----
    dc = getattr(gd, a.dataconfig)
    ref = ReferenceSet.from_dataconfigs(dc)
    recs = list(itertools.islice(
        _stream_records(dc, dict(Curriculum().params(a.progress)), seed=777001), a.probe_size))
    probe_in, probe_tg = build_batch(recs, ref, cfg, dev)

    def task_eval(m, inp, _tg=probe_tg, _cfg=cfg):
        return eval_metrics(m(inp), _tg, _cfg)

    def task_losses(m, inp, _tg=probe_tg, _cfg=cfg, _lv=logvars):
        return hierarchical_loss(m(inp), _tg, _cfg, _lv)[1]

    xr = ModelXRay(model, lr=3e-4, deep_every=1, uncertainty=logvars,
                   task_eval=task_eval, task_losses=task_losses, shared_module="embedding")

    # ---- live tier: one full deep observe (needs grads on the probe loss) ----
    model.train()
    total, parts = hierarchical_loss(model(probe_in), probe_tg, cfg, logvars)
    model.zero_grad(); total.backward()
    live = xr.observe(step=int(ck.get("step", 0)), loss=float(total.detach()),
                      parts={k: float(v) for k, v in parts.items()}, probe_input=probe_in)
    model.zero_grad()

    # ---- capture V-head penultimate features + logits (neural collapse + margins) ----
    grab = {}
    h = model.branches["v"].cls_head.register_forward_hook(
        lambda mod, inp, out: grab.update(feat=inp[0].detach(), logits=out.detach()))
    with torch.no_grad():
        model.eval(); model(probe_in)
    h.remove()
    v_labels = probe_tg["v_allele"].argmax(-1).to(dev)          # true top-1 V class per read

    def loss_closure():
        return hierarchical_loss(model(probe_in), probe_tg, cfg, logvars)[0]

    heavy = xr.deep_report(probe_input=probe_in, loss_closure=loss_closure,
                           nc_features=grab["feat"], nc_labels=v_labels,
                           margin_logits=grab["logits"], margin_labels=v_labels, hessian_iters=20)

    # ---- report ----
    print("\n" + "=" * 68); print(f"ALIGNAIR DIAGNOSIS  @ step {ck.get('step')}"); print("=" * 68)
    ev = live["eval"]
    print("\n-- PER-TASK QUALITY (held-out probe, progress={}) --".format(a.progress))
    print(f"  V={ev['v_allele_top1']:.3f}  D={ev['d_allele_top1']:.3f}  J={ev['j_allele_top1']:.3f}  "
          f"orientation={ev['orientation_acc']:.3f}")
    print(f"  seg MAE  v={ev['v_seg_mae']:.2f} d={ev['d_seg_mae']:.2f} j={ev['j_seg_mae']:.2f} nt   "
          f"mutation MAE={ev['mutation_mae']:.3f}  indel MAE={ev['indel_mae']:.2f}  prod={ev['productive_acc']:.3f}")

    print("\n-- GENERALIZATION: weightwatcher power-law alpha (2<=a<=6 healthy; <2 over-/ >6 under-trained) --")
    for name, al in sorted(heavy["weightwatcher_alpha"].items()):
        flag = "" if 2 <= al <= 6 else "  <-- outside healthy band"
        print(f"  {name:<34} alpha={al:.2f}{flag}")

    print("\n-- LOSS-SURFACE SHARPNESS (flatter = better generalization) --")
    print(f"  Hessian top-eigenvalue = {heavy['hessian_top_eig']:.3e}   trace = {heavy['hessian_trace']:.3e}")

    print("\n-- NEURAL COLLAPSE (V head; lower NC1 = tighter class clusters) --")
    nc = heavy["neural_collapse"]
    print(f"  NC1 within/between = {nc['nc1_within_between']:.3f}   "
          f"NC2 (equinorm cv) = {nc.get('nc2_equinorm', float('nan')):.3f}   "
          f"NC3 = {nc.get('nc3_self_dual', float('nan')):.3f}")
    mg = heavy["margin"]
    print(f"  V classification margin: mean={mg['mean']:.2f}  (frac<0 = misclassified = {mg.get('frac_negative', float('nan')):.3f})")

    print("\n-- FEATURE GEOMETRY (per layer: effective rank / anisotropy) --")
    for k, g in live["geometry"].items():
        print(f"  {k:<22} effR {g['eff_rank']:6.1f}/{g['dim']:<4} intrinsic {g['intrinsic_dim']:5.1f} "
              f"aniso {g['anisotropy']:.2f} collin {g['collinearity']:.2f}")

    print("\n-- MULTI-TASK GRADIENT CONFLICT (shared embedding; neg = tasks fighting) --")
    for pair, c in sorted(live["interference"]["cosine"].items(), key=lambda x: x[1])[:6]:
        print(f"  {pair:<26} {c:+.2f}")

    print("\n-- KENDALL TASK WEIGHTS (exp(-logvar); higher = model trusts it more) --")
    print("  " + "  ".join(f"{k}={v['weight']:.1f}" for k, v in live["uncertainty"].items()))

    print(f"\n-- HEALTH FLAGS -- {live['flags'] or 'none (clean)'}")

    out = a.out or (a.ckpt[:-3] if a.ckpt.endswith(".pt") else a.ckpt) + ".diagnosis.json"
    with open(out, "w") as f:
        json.dump({"step": ck.get("step"), "eval": ev, "heavy": heavy,
                   "geometry": live["geometry"], "interference": live["interference"],
                   "uncertainty": live["uncertainty"], "flags": live["flags"]}, f, indent=2, default=float)
    print(f"\nfull diagnosis JSON -> {out}")


if __name__ == "__main__":
    main()
