"""Proven error decomposition for the AA IGH model.

For each full-length V read: compare the trained model's V call against the AA-nearest-germline
ORACLE (the best any classifier can do from protein alone). Every V error is then provably one of:
  - info-ceiling:      oracle ALSO misses -> the true allele is not recoverable from the AA (loss)
  - model-deficiency:  oracle recovers it, model does not             -> fixable by training/capacity
Both judged at protein-CLASS level (synonymous alleles are indistinguishable by construction).
Bucketed by SHM so we can see whether mutation lowers the *ceiling* (loss) or only the model.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/error_forensics.py --device cuda --n-per-stratum 1200
"""
import argparse
import itertools
import json
import os
from collections import defaultdict

import numpy as np
import torch

from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.data import build_class_maps
from alignair.aa.forensics import germline_id_matrix, nearest_germline
from alignair.aa.gym_bridge import aa_usable, build_batch_aa
from alignair.aa.reference import build_aa_germlines
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.aa.train_gym import setup_aa


def _shm_bucket(r):
    return "0-5%" if r < 0.05 else "5-10%" if r < 0.10 else "10-15%" if r < 0.15 else "15%+"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/same_scheme/model.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--min-v-residues", type=int, default=20)
    ap.add_argument("--out", default="experiments/aa/same_scheme/forensics.json")
    args = ap.parse_args()

    model, ref, cfg, _ = setup_aa(args.dc, 192)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()

    cmaps, _ = build_class_maps(args.dc)
    vclass = cmaps["V"].allele_class                              # allele name -> protein class id
    gset = build_aa_germlines(getattr(__import__("GenAIRR"), args.dc), genes=("V",))
    names, G, glen, ganchor = germline_id_matrix(ref, gset.genes["V"], "V")
    idx_class = np.array([vclass.get(n, -1) for n in names])      # class id per allele index
    tok = AAProteinTokenizer()

    # collect full-length V reads (V starts at germline start; prefix-anchored oracle is valid)
    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    cases = []
    for c in stream_benchmark(spec, dataconfig=None, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not aa_usable(r):
            continue
        cases.append((c.stratum, r))                             # Cys-anchoring handles 5' trim/frame

    rows = []
    with torch.no_grad():
        for i in range(0, len(cases), 256):
            chunk = [r for _, r in cases[i:i + 256]]
            strata = [s for s, _ in cases[i:i + 256]]
            bi, tg = build_batch_aa(chunk, ref, cfg, args.device)
            probs = model(bi)["v_allele"].cpu().numpy()
            for k, rec in enumerate(chunk):
                # Cys anchor residue in the observed protein = junction_start // 3; V is everything
                # before it. Anchor-align to each germline (robust to frame offset / 5' trim).
                obs_cys = int(rec["junction_start"]) // 3
                if obs_cys < args.min_v_residues:
                    continue
                obs = tok.encode(rec["sequence_aa"])
                oracle_idx, ident = nearest_germline(obs, obs_cys, G, ganchor)
                true_idxs = [j for j in (tg["v_allele"][k] > 0.5).nonzero().flatten().tolist()]
                if not true_idxs:
                    continue
                true_classes = {int(idx_class[j]) for j in true_idxs}
                model_idx = int(probs[k].argmax())
                # rank of the best true allele in the model's probability ordering
                order = np.argsort(-probs[k])
                true_rank = int(min(np.where(order == j)[0][0] for j in true_idxs)) + 1

                rows.append({
                    "stratum": strata[k],
                    "shm": float(rec["mutation_rate"]),
                    "shm_bucket": _shm_bucket(float(rec["mutation_rate"])),
                    "v_residues": obs_cys,
                    "has_indel": int(rec.get("n_indels", 0)) > 0,
                    "true_ident": float(max(ident[j] for j in true_idxs)),   # sanity: obs vs true germline
                    "model_ok": int(int(idx_class[model_idx]) in true_classes),
                    "oracle_ok": int(int(idx_class[oracle_idx]) in true_classes),
                    "true_rank": true_rank,
                })

    # The prefix-anchored AA-identity oracle is only valid for indel-free reads (an indel frameshifts
    # the protein downstream, breaking the germline correspondence). Decompose errors on indel-free
    # reads; report indel reads separately (their frameshift is a distinct, provable failure mode).
    indel_rows = [r for r in rows if r["has_indel"]]
    rows = [r for r in rows if not r["has_indel"]]
    n = len(rows)

    def acc(sub, key):
        return float(np.mean([r[key] for r in sub])) if sub else 0.0

    # overall
    model_acc, oracle_acc = acc(rows, "model_ok"), acc(rows, "oracle_ok")
    err = [r for r in rows if not r["model_ok"]]
    ceiling = [r for r in err if not r["oracle_ok"]]          # oracle also wrong -> info loss
    deficiency = [r for r in err if r["oracle_ok"]]           # oracle right, model wrong -> fixable
    beats = [r for r in rows if r["model_ok"] and not r["oracle_ok"]]

    def by(field, keys=None):
        g = defaultdict(list)
        for r in rows:
            g[r[field]].append(r)
        keys = keys or sorted(g)
        return {kk: {"n": len(g[kk]), "oracle_acc": acc(g[kk], "oracle_ok"),
                     "model_acc": acc(g[kk], "model_ok"),
                     "mean_true_ident": acc(g[kk], "true_ident")} for kk in keys if kk in g}

    report = {
        "n_reads": n,
        "overall": {"model_class_acc": model_acc, "oracle_class_acc": oracle_acc,
                    "model_below_ceiling": oracle_acc - model_acc},
        "error_decomposition": {
            "n_model_errors": len(err),
            "info_ceiling_frac": len(ceiling) / max(1, len(err)),
            "model_deficiency_frac": len(deficiency) / max(1, len(err)),
            "model_beats_oracle_frac": len(beats) / max(1, n)},
        "true_rank_when_wrong": {
            "median": float(np.median([r["true_rank"] for r in err])) if err else 0.0,
            "frac_rank_le_3": float(np.mean([r["true_rank"] <= 3 for r in err])) if err else 0.0,
            "frac_rank_le_10": float(np.mean([r["true_rank"] <= 10 for r in err])) if err else 0.0},
        "by_shm": by("shm_bucket", ["0-5%", "5-10%", "10-15%", "15%+"]),
        "by_stratum": by("stratum"),
        "indel_reads": {"n": len(indel_rows),
                        "model_acc": acc(indel_rows, "model_ok"),
                        "mean_true_ident": acc(indel_rows, "true_ident"),
                        "note": "prefix-anchored oracle invalid under frameshift; oracle_acc omitted"},
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n==== AA V-allele ERROR FORENSICS (n={n} indel-free full-length V reads; "
          f"{len(indel_rows)} indel reads analyzed separately) ====")
    print(f"MODEL class acc {model_acc:.3f}   vs   ORACLE (AA ceiling) {oracle_acc:.3f}   "
          f"-> model is {oracle_acc - model_acc:.3f} below the achievable ceiling")
    ed = report["error_decomposition"]
    print(f"\nof {ed['n_model_errors']} model errors:")
    print(f"  {ed['info_ceiling_frac']:.1%} are INFO-CEILING (oracle also fails -> unrecoverable from AA)")
    print(f"  {ed['model_deficiency_frac']:.1%} are MODEL-DEFICIENCY (oracle succeeds -> fixable)")
    tr = report["true_rank_when_wrong"]
    print(f"  when wrong, true allele median rank {tr['median']:.0f}  "
          f"(<=3: {tr['frac_rank_le_3']:.1%}, <=10: {tr['frac_rank_le_10']:.1%})")
    print(f"\n{'SHM':>8} {'n':>6} {'oracle':>7} {'model':>7} {'obs~true':>9}")
    for b, d in report["by_shm"].items():
        print(f"{b:>8} {d['n']:6d} {d['oracle_acc']:7.3f} {d['model_acc']:7.3f} {d['mean_true_ident']:9.3f}")
    print(f"\n{'stratum':22s} {'n':>6} {'oracle':>7} {'model':>7}")
    for s, d in sorted(report["by_stratum"].items(), key=lambda kv: kv[1]["oracle_acc"], reverse=True):
        print(f"{s:22s} {d['n']:6d} {d['oracle_acc']:7.3f} {d['model_acc']:7.3f}")
    ir = report["indel_reads"]
    print(f"\nINDEL reads (frameshift; oracle N/A): n={ir['n']}  model_acc {ir['model_acc']:.3f}  "
          f"obs~true {ir['mean_true_ident']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
