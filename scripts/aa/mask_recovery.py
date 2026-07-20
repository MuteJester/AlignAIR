"""How much of the INDEL damage to V is recoverable by MASKING vs by frame CORRECTION?

Measured robustly from the V CIGAR (no fragile frame-guessing):
  - an indel splits the V alignment into contiguous match (M) runs;
  - MASKING (keep only the valid pre-frameshift prefix) recovers the FIRST M-run;
  - full frame CORRECTION (re-frame at each indel) recovers ALL M-runs (total M).
Split by indel location: an indel downstream of V (D/J) leaves V's frame intact (V undamaged);
an indel inside V frameshifts it. Also report the model's current V accuracy per split.

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
from alignair.aa.train_gym import setup_aa


def m_runs_nt(cigar: str):
    """Contiguous match-run lengths (nt) of a V CIGAR, split by internal indels; leading/trailing
    germline trims (D with no read consumption at the ends) do not create spurious runs."""
    runs, cur = [], 0
    for n, op in re.findall(r"(\d+)([MID])", cigar or ""):
        n = int(n)
        if op == "M":
            cur += n
        else:                                    # I or D breaks the in-frame run
            if cur:
                runs.append(cur)
                cur = 0
    if cur:
        runs.append(cur)
    return runs


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
                if call not in vclass:
                    continue
                true_c = vclass[call]
                model_ok = int(int(idx_class[int(probs[k].argmax())]) == true_c)
                runs = m_runs_nt(rec.get("v_cigar", ""))
                row = {"model_ok": model_ok,
                       "maskable_res": (runs[0] // 3) if runs else 0,       # first in-frame run
                       "correctable_res": sum(runs) // 3}                   # all match runs
                (dj if int(rec.get("n_v_indels", 0)) == 0 else vind).append(row)

    def frac(rows, pred):
        return float(np.mean([pred(r) for r in rows])) if rows else 0.0

    def mean(rows, key):
        return float(np.mean([r[key] for r in rows])) if rows else 0.0

    mc = args.min_call_residues
    report = {
        "n_indel_reads": len(dj) + len(vind), "min_call_residues": mc,
        "dj_indels": {"n": len(dj), "frac_of_indel_reads": len(dj) / max(1, len(dj) + len(vind)),
                      "model_v_acc": mean(dj, "model_ok"),
                      "mean_correctable_res": mean(dj, "correctable_res")},
        "v_indels": {"n": len(vind), "frac_of_indel_reads": len(vind) / max(1, len(dj) + len(vind)),
                     "model_v_acc": mean(vind, "model_ok"),
                     "mean_maskable_res": mean(vind, "maskable_res"),
                     "mean_correctable_res": mean(vind, "correctable_res"),
                     "frac_maskable_ge_min": frac(vind, lambda r: r["maskable_res"] >= mc),
                     "frac_needs_correction": frac(
                         vind, lambda r: r["maskable_res"] < mc <= r["correctable_res"]),
                     "frac_v_truly_lost": frac(vind, lambda r: r["correctable_res"] < mc)},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    d, v = report["dj_indels"], report["v_indels"]
    tot = report["n_indel_reads"]
    print(f"\n==== INDEL DAMAGE TO V: what MASKING vs CORRECTION can recover (n={tot} indel reads) ====")
    print(f"\n[{d['frac_of_indel_reads']:.0%}] indel is DOWNSTREAM of V (D/J) -> V frame INTACT ({d['n']} reads)")
    print(f"    V undamaged; model currently gets {d['model_v_acc']:.3f} (a model gap, not a masking one)")
    print(f"\n[{v['frac_of_indel_reads']:.0%}] indel is INSIDE V -> V frameshifted ({v['n']} reads)")
    print(f"    model currently gets {v['model_v_acc']:.3f} on the frameshifted input")
    print(f"    valid V residues -- MASKING (first run): {v['mean_maskable_res']:.0f}   "
          f"CORRECTION (all runs): {v['mean_correctable_res']:.0f}")
    print(f"    of these V-indel reads (>= {mc} residues to call V):")
    print(f"       MASKING alone suffices:            {v['frac_maskable_ge_min']:.0%}")
    print(f"       needs full frame CORRECTION:       {v['frac_needs_correction']:.0%}")
    print(f"       V truly lost (too little even so): {v['frac_v_truly_lost']:.0%}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
