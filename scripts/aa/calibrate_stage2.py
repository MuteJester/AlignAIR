"""Stage-2: conformal calibration of the frozen Stage-1 model into calibrated V/D/J allele SETS.

Distinct CALIBRATION and untouched TEST cohorts (disjoint gym seeds), each a coverage mix (full-length
+ truncated). APS at a declared 95% target, per gene, with Mondrian (coverage-bin) calibration +
pooled fallback. Reports empirical coverage (+Wilson CI), set size (mean/median/95th), singleton rate,
by coverage stratum. V/J protein-class sets are expanded to nucleotide-allele sets (the output contract).

Run: PYTHONPATH=src .venv/bin/python scripts/aa/calibrate_stage2.py --model experiments/aa/strategic/model_stage1_PASS.pt --device cuda
"""
import argparse
import hashlib
import json
import os

import numpy as np
import torch

import GenAIRR as ga
from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.augment import aa_truncate
from alignair.aa.conformal import Conformal
from alignair.aa.data import build_class_maps
from alignair.aa.evaluate import wilson_interval
from alignair.aa.gym_bridge import aa_usable
from alignair.aa.strategic import strategic_setup
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.reference.reference_set import ReferenceSet


def _cohort(dc, ref, vclass, jclass, d_index, n, seed, rng):
    rows = []
    for c in stream_benchmark(default_igh_spec(n_per_stratum=n, seed=seed), dataconfig=dc, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not aa_usable(r):
            continue
        vc, jc = r["v_call"].split(",")[0], r["j_call"].split(",")[0]
        dc0 = (r.get("d_call") or "").split(",")[0].strip()
        if vc not in vclass or jc not in jclass:
            continue
        aa = r["sequence_aa"]
        cov = "full"
        if rng.random() < 0.5:                                # coverage mix: half truncated
            keep = float(rng.uniform(0.3, 0.9))
            aa = aa_truncate(aa, keep, keep, rng)
            cov = "short" if keep < 0.6 else "trunc"
        rows.append({"aa": aa, "v": vclass[vc], "j": jclass[jc],
                     "d": d_index.get(dc0, -1) if not r.get("d_inverted") else -1, "cov": cov})
        if len(rows) >= n:
            break
    return rows


def _probs(model, seqs, cfg, device, key, batch=256):
    tok = AAProteinTokenizer()
    L = cfg.max_seq_length
    out = []
    for i in range(0, len(seqs), batch):
        chunk = seqs[i:i + batch]
        toks = torch.zeros(len(chunk), L, dtype=torch.long)
        for j, s in enumerate(chunk):
            e = tok.encode(s)
            n = min(len(e), L)
            toks[j, :n] = torch.from_numpy(e[:n])
        with torch.no_grad():
            lg = model({"tokenized_sequence": toks.to(device)})[key].cpu()
        out.append(torch.softmax(lg, -1).numpy())
    return np.concatenate(out, 0)


def _set_stats(sizes):
    sizes = np.asarray(sizes)
    return {"mean": float(sizes.mean()), "median": float(np.median(sizes)),
            "p95": float(np.percentile(sizes, 95)), "singleton_rate": float((sizes == 1).mean())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/strategic/model_stage1_PASS.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--alpha", type=float, default=0.05)      # 95% target
    ap.add_argument("--n", type=int, default=5000)            # per cohort
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/superpowers/artifacts/2026-07-20-aa-strategic/calibration.json")
    args = ap.parse_args()

    model, cfg, class_maps, meta = strategic_setup(args.dc, 192, has_d=True)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()
    dc = getattr(ga, args.dc)
    ref = ReferenceSet.from_dataconfigs(dc)
    vclass, jclass = class_maps["V"].allele_class, class_maps["J"].allele_class
    d_index = meta["d_index"]
    d_name = {i: n for n, i in d_index.items()}

    cal = _cohort(dc, ref, vclass, jclass, d_index, args.n, seed=1000, rng=np.random.default_rng(1))
    tst = _cohort(dc, ref, vclass, jclass, d_index, args.n, seed=9000, rng=np.random.default_rng(2))
    print(f"calibration {len(cal)} / test {len(tst)} (disjoint seeds); target coverage {1 - args.alpha:.0%}")

    h = hashlib.sha256()
    with open(args.model, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    report = {"model": args.model, "alpha": args.alpha, "n_cal": len(cal), "n_test": len(tst),
              "provenance": {
                  "model_sha256": h.hexdigest(),
                  "class_map_db_hash": {"V": class_maps["V"].db_hash, "J": class_maps["J"].db_hash},
                  "target_coverage": 1 - args.alpha,
                  "bin_defs": "coverage groups: full=uncropped; trunc=kept∈[0.6,0.9]; short=kept∈[0.3,0.6)",
                  "min_group_fallback": 500,
                  "supported_distribution": ("SIMULATOR-DOMAIN ONLY: GenAIRR HUMAN_IGH_OGRDB, coverage-mix "
                                             "(full+truncated). Guarantee holds under exchangeability within "
                                             "each Mondrian group; NOT validated on real protein — recalibrate "
                                             "before real-data use."),
              }, "genes": {}}
    for gene, key, cmap in (("V", "v_class_logits", class_maps["V"]),
                            ("J", "j_class_logits", class_maps["J"]),
                            ("D", "d_class_logits", None)):
        gk = gene.lower()
        cal_valid = [r for r in cal if r[gk] >= 0]
        tst_valid = [r for r in tst if r[gk] >= 0]
        p_cal = _probs(model, [r["aa"] for r in cal_valid], cfg, args.device, key)
        p_tst = _probs(model, [r["aa"] for r in tst_valid], cfg, args.device, key)
        y_cal = np.array([r[gk] for r in cal_valid])
        cov_cal = np.array([r["cov"] for r in cal_valid])
        conf = Conformal(alpha=args.alpha, min_group=500, seed=0).fit(p_cal, y_cal, groups=cov_cal)

        sizes, nt_sizes, covered, by_cov = [], [], [], {}
        for i, r in enumerate(tst_valid):
            s = conf.predict_set(p_tst[i], group=r["cov"])
            sizes.append(len(s))
            covered.append(r[gk] in s)
            by_cov.setdefault(r["cov"], []).append((len(s), r[gk] in s))
            if gene in ("V", "J"):                            # expand protein classes -> nt alleles
                nt = set().union(*[cmap.class_members[c] for c in s]) if s else set()
                nt_sizes.append(len(nt))
        n = len(tst_valid)
        k = int(np.sum(covered))
        g = {"n_test": n, "empirical_coverage": k / n, "coverage_ci95": list(wilson_interval(k, n)),
             "set_size": _set_stats(sizes),
             "by_coverage_stratum": {cv: {"n": len(v), "coverage": float(np.mean([x[1] for x in v])),
                                          "mean_set": float(np.mean([x[0] for x in v]))}
                                     for cv, v in by_cov.items()},
             "calibration_frozen": {                          # the deployable quantiles (research snapshot)
                 "q_pooled": conf.q_pooled,
                 "q_mondrian": {str(k): float(vq) for k, vq in conf.q_group.items()},
                 "groups_fitted": list(conf.q_group.keys()),
                 "pooled_fallback_exercised": len(conf.q_group) < len(set(cov_cal))}}
        if nt_sizes:
            g["nt_allele_set_size"] = _set_stats(nt_sizes)
        report["genes"][gene] = g

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n==== Stage-2 conformal calibration (target {1 - args.alpha:.0%}) ====")
    for gene, g in report["genes"].items():
        ci = g["coverage_ci95"]
        line = (f"  {gene}: coverage {g['empirical_coverage']:.3f} CI[{ci[0]:.3f},{ci[1]:.3f}]  "
                f"set(mean {g['set_size']['mean']:.2f}, median {g['set_size']['median']:.0f}, "
                f"p95 {g['set_size']['p95']:.0f}, singleton {g['set_size']['singleton_rate']:.2f})")
        if "nt_allele_set_size" in g:
            line += f"  nt-alleles mean {g['nt_allele_set_size']['mean']:.2f}"
        print(line)
        for cv, s in g["by_coverage_stratum"].items():
            print(f"      [{cv:5s}] n={s['n']:4d} coverage {s['coverage']:.3f} mean-set {s['mean_set']:.2f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
