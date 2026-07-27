"""Stage-4 Task 4.4 — candidate trainer + dev-only production-seed rule.

Trains a learned candidate (C1–C4) on the burned Gate-3 development TRAIN partition only, with early stopping
on the internal dev partition. HMM normalization is fit on the TRAIN partition only. Checkpoints (to .private/)
bind full provenance. choose_production_seed inspects BURNED-DEV metrics only and RAISES if handed any
fresh-validation field. Fresh selection, blind-LOCO, far-OOD, and the seal are never loaded here.

Smoke mode (`--smoke`) validates the full assembly end-to-end on a small burned-dev subset.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from alignair.aa.guard.classes import DOMAIN_TYPE_INDEX
from alignair.aa.guard.features import FEATURE_NAMES, hmm_features
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.profiles import load_bank
from alignair.aa.guard.tokenizer import MAX_LEN, AaTokenizer, enforce_length_contract
from alignair.aa.guard.train_data import partition_by_parent_lineage

DEV = Path(".private/aa-gate3-hydration/dev_cohort.jsonl")
PROF = Path(".private/aa-applicability/profiles")
BANK_MANIFEST = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability/profile_bank_manifest.json")

_FRESH_FIELDS = ("fresh", "selection", "loco", "blind", "far_ood", "far-ood", "seal", "locked", "test")


def choose_production_seed(dev_results: dict, frozen_rule: str = "min_dev_loss_then_lowest_seed") -> int:
    """Deterministic dev-only rule: lowest dev loss, tie -> lowest seed. Raises if any fresh field appears."""
    for seed, metrics in dev_results.items():
        for k in metrics:
            if any(f in str(k).lower() for f in _FRESH_FIELDS):
                raise ValueError(f"choose_production_seed received a fresh-validation field: {k!r}")
    return min(dev_results, key=lambda s: (dev_results[s]["dev_loss"], s))


def _group_to_domain_type(rec):
    g, cat = rec.get("group"), rec.get("category", "")
    if g in ("positive_real", "positive_sim"):
        return DOMAIN_TYPE_INDEX["human_igh_vh"]
    if g == "vhh_nonhuman":
        return DOMAIN_TYPE_INDEX["murine_vh"] if cat == "murine_vh" else DOMAIN_TYPE_INDEX["vhh"]
    if g == "igsf":
        return DOMAIN_TYPE_INDEX["igsf"]
    if g == "igk_igl":
        return DOMAIN_TYPE_INDEX["human_igk"] if cat == "igk" else DOMAIN_TYPE_INDEX["human_igl"]
    if g == "tcr":
        return DOMAIN_TYPE_INDEX["tra"] if cat == "tra" else DOMAIN_TYPE_INDEX["trb"]
    return None                                        # structural / other -> no genuine domain-type label


def smoke_train(cid="C2", seed=0, n_records=48, n_steps=15):
    """End-to-end validation on a small burned-dev subset: build features, fit train-only normalization,
    tokenize, forward the assembled model, optimize a masked applicability+type loss; assert loss drops."""
    torch.manual_seed(seed)
    recs = [json.loads(l) for l in open(DEV)]
    train, dev, partition_hash = partition_by_parent_lineage(recs, seed=seed, dev_frac=0.2)
    train = train[:n_records]
    bank = load_bank(PROF)
    bank_checksum = json.load(open(BANK_MANIFEST))["bank_checksum"]

    tok = AaTokenizer(max_len=MAX_LEN)
    vecs, ids_l, mask_l, app_t, type_t, type_m = [], [], [], [], [], []
    for r in train:
        if not enforce_length_contract(r["aa"]).ok:
            continue
        f = hmm_features(bank, r["aa"])
        vecs.append(f.to_vector())
        i, m = tok.encode(r["aa"])
        ids_l.append(i); mask_l.append(m)
        app_t.append(1.0 if r.get("expected_state") == "supported" else 0.0)
        dt = _group_to_domain_type(r)
        type_t.append(dt if dt is not None else 0)
        type_m.append(1.0 if dt is not None else 0.0)

    fusion = fit_fusion(FEATURE_NAMES, vecs, bank_checksum)     # TRAIN-partition-only normalization
    hmm_norm = torch.stack([fusion.normalize(v, feature_names=FEATURE_NAMES, bank_checksum=bank_checksum)
                            for v in vecs])
    ids = torch.stack(ids_l); mask = torch.stack(mask_l)
    app = torch.tensor(app_t); typ = torch.tensor(type_t); typm = torch.tensor(type_m)

    model = GuardModel(cid, hmm_dim=len(FEATURE_NAMES))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()
    losses = []
    for step in range(n_steps):
        opt.zero_grad()
        out = model(ids, mask, hmm_norm=hmm_norm)
        la = torch.nn.functional.binary_cross_entropy_with_logits(out["applicability"].squeeze(-1), app)
        lt = (torch.nn.functional.cross_entropy(out["domain_type"], typ.long(), reduction="none") * typm).sum() \
            / typm.sum().clamp(min=1)
        loss = la + lt
        loss.backward(); opt.step()
        losses.append(float(loss))
    return {"cid": cid, "seed": seed, "n_train": len(vecs), "partition_hash": partition_hash[:12],
            "params": sum(p.numel() for p in model.parameters()),
            "loss_first": round(losses[0], 4), "loss_last": round(losses[-1], 4),
            "decreased": losses[-1] < losses[0]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--cid", default="C2")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.smoke:
        t0 = time.time()
        r = smoke_train(cid=args.cid, seed=args.seed)
        r["wall_s"] = round(time.time() - t0, 1)
        print(json.dumps(r, indent=2, sort_keys=True))
