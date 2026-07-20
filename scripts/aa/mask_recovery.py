"""How much INDEL damage to V is recoverable by MASKING vs frame CORRECTION? (corrected 2026-07-20)

CORRECTION over the earlier version: GenAIRR builds `sequence_aa` in the frame anchored at the V-Cys
(`junction_start % 3`). So for an indel *before* the Cys, the ANCHOR-PROXIMAL (near-Cys) region stays
in-frame and the 5' region goes off-frame — the OPPOSITE of "keep the first match run". The maskable
valid region is therefore the contiguous in-frame run adjacent to the Cys (measured anchor-relative
against the true germline). Verified: near-Cys identity ~0.64 vs 5'-most ~0.22 on single-V-indel reads.

Two recovery notions, complete-codon counted:
  - MASKING: keep the anchor-proximal in-frame run only (residues from the Cys back to the first indel).
  - CORRECTION: re-frame at each indel and keep all match runs (sum of complete codons per CIGAR run).

NOTE: `maskable` here uses the true germline + true junction to locate the frameshift — it is a
HIDDEN-TRUTH diagnostic (a ceiling for what a perfect mask could keep), NOT an AA-only method. The
no-training local aligner (de-risk experiment) does the AA-only version by scanning Cys candidates.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/mask_recovery.py --device cuda --n-per-stratum 1500
"""
import argparse
import json
import os
import re

import numpy as np
import torch

from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.data import build_class_maps
from alignair.aa.gym_bridge import aa_usable, build_batch_aa
from alignair.aa.reference import build_aa_germlines
from alignair.aa.train_gym import setup_aa


def _m_runs_nt(cigar: str):
    runs, cur = [], 0
    for n, op in re.findall(r"(\d+)([MID])", cigar or ""):
        n = int(n)
        if op == "M":
            cur += n
        else:
            if cur:
                runs.append(cur)
                cur = 0
    if cur:
        runs.append(cur)
    return runs


def _anchor_proximal_run(match_from5: list, win: int = 5) -> int:
    """Length of the in-frame run adjacent to the Cys anchor: scan 5' FROM the anchor (reversed),
    stop at the first window with <=1 match (the frameshift boundary)."""
    rev = match_from5[::-1]                        # index 0 = anchor-proximal residue
    for i in range(len(rev) - win + 1):
        if sum(rev[i:i + win]) <= 1:
            return i
    return len(rev)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/same_scheme/model.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-call-residues", type=int, default=25)
    ap.add_argument("--out", default="experiments/aa/same_scheme/mask_recovery.json")
    args = ap.parse_args()

    model, ref, cfg, _ = setup_aa(args.dc, 192)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()
    vclass = build_class_maps(args.dc)[0]["V"].allele_class
    germ = {n: g for n, g in build_aa_germlines(getattr(__import__("GenAIRR"), args.dc),
                                                genes=("V",)).genes["V"].items()}
    names = ref.gene("V").names
    idx_class = np.array([vclass.get(n, -1) for n in names])

    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    cases = [c.record for c in stream_benchmark(spec, dataconfig=None, reference_set=ref)
             if c.orientation_id == 0 and aa_usable(c.record) and int(c.record.get("n_indels", 0)) > 0]

    dj, vind = [], []
    with torch.no_grad():
        for i in range(0, len(cases), 256):
            chunk = cases[i:i + 256]
            probs = model(build_batch_aa(chunk, ref, cfg, args.device)[0])["v_allele"].cpu().numpy()
            for k, rec in enumerate(chunk):
                call = rec["v_call"].split(",")[0]
                if call not in vclass or call not in germ:
                    continue
                true_c = vclass[call]
                model_ok = int(int(idx_class[int(probs[k].argmax())]) == true_c)
                correctable = sum(r // 3 for r in _m_runs_nt(rec.get("v_cigar", "")))

                if int(rec.get("n_v_indels", 0)) == 0:
                    dj.append({"model_ok": model_ok, "correctable_res": correctable})
                    continue
                # anchor-proximal maskable run (hidden-truth): Cys-anchor vs true germline
                g = germ[call]
                obs_cys = int(rec["junction_start"]) // 3
                t = min(obs_cys, g.anchor_aa)
                aa = rec["sequence_aa"]
                o, gg = aa[obs_cys - t:obs_cys], g.aa_seq[g.anchor_aa - t:g.anchor_aa]
                L = min(len(o), len(gg))
                maskable = _anchor_proximal_run([o[j] == gg[j] for j in range(L)]) if L else 0
                vind.append({"model_ok": model_ok, "maskable_res": maskable,
                             "correctable_res": correctable})

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    def frac(rows, pred):
        return float(np.mean([pred(r) for r in rows])) if rows else 0.0

    mc = args.min_call_residues
    tot = len(dj) + len(vind)
    report = {
        "n_indel_reads": tot, "min_call_residues": mc,
        "note": "maskable = anchor-proximal in-frame run (near Cys); hidden-truth diagnostic, not AA-only",
        "dj_indels": {"n": len(dj), "frac": len(dj) / max(1, tot),
                      "model_v_acc": mean(dj, "model_ok"), "mean_correctable_res": mean(dj, "correctable_res")},
        "v_indels": {"n": len(vind), "frac": len(vind) / max(1, tot),
                     "model_v_acc": mean(vind, "model_ok"),
                     "mean_maskable_res": mean(vind, "maskable_res"),
                     "mean_correctable_res": mean(vind, "correctable_res"),
                     "frac_maskable_ge_min": frac(vind, lambda r: r["maskable_res"] >= mc),
                     "frac_needs_correction": frac(vind, lambda r: r["maskable_res"] < mc <= r["correctable_res"]),
                     "frac_truly_lost": frac(vind, lambda r: r["correctable_res"] < mc)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    d, v = report["dj_indels"], report["v_indels"]
    print(f"\n==== INDEL DAMAGE TO V (CORRECTED anchor-relative) — n={tot} indel reads ====")
    print(f"\n[{d['frac']:.0%}] indel DOWNSTREAM of V (D/J) -> V frame intact ({d['n']} reads); "
          f"model V acc {d['model_v_acc']:.3f}")
    print(f"\n[{v['frac']:.0%}] indel INSIDE V ({v['n']} reads); model V acc {v['model_v_acc']:.3f}")
    print(f"    anchor-proximal MASKABLE V residues: {v['mean_maskable_res']:.0f}   "
          f"CORRECTABLE (all runs): {v['mean_correctable_res']:.0f}")
    print(f"    of V-indel reads (>= {mc} residues to call V):")
    print(f"       MASKING (anchor-proximal) suffices: {v['frac_maskable_ge_min']:.0%}")
    print(f"       needs full frame CORRECTION:        {v['frac_needs_correction']:.0%}")
    print(f"       V truly lost:                       {v['frac_truly_lost']:.0%}")
    print(f"\n(maskable is a HIDDEN-TRUTH ceiling; the AA-only recovery number comes from the "
          f"no-training local aligner)")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
