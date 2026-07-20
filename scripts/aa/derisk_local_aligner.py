"""De-risk (a): does a NO-TRAINING, pure-AA local aligner recover V better than the trained neural
head — enough to justify building an alignment re-ranker into the strategic AA model?

Predeclared decision rule (set before seeing results):
  - PROCEED with the re-ranker iff the aligner beats the neural head by >= 5 points on V-indel reads
    AND loses <= 2 points on clean reads.
The aligner is pure AA (no true junction, no CIGAR); local alignment scans the anchor implicitly.
Hidden-CIGAR maskable/correctable is reported as a DIAGNOSTIC only, never as an AA-only baseline.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/derisk_local_aligner.py --device cuda --n-per-stratum 1500
"""
import argparse
import json
import os
import re
from collections import defaultdict

import numpy as np
import torch

from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.data import build_class_maps
from alignair.aa.local_aligner import classify_v, v_class_prototypes
from alignair.aa.reference import build_aa_germlines
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.aa.train_gym import setup_aa

_AA = "ACDEFGHIKLMNPQRSTVWY"


def _aa_substitute(seq, rate, rng):
    s = list(seq)
    for i, ch in enumerate(s):
        if ch in _AA and rng.random() < rate:
            s[i] = _AA[int(rng.integers(20))]
    return "".join(s)


def _aa_indel(seq, rng):
    i = int(rng.integers(max(1, len(seq))))
    if rng.random() < 0.5:
        return seq[:i] + seq[i + 1:]                       # single-residue deletion
    return seq[:i] + _AA[int(rng.integers(20))] + seq[i:]  # single-residue insertion


def _correctable_res(cigar):
    return sum(int(n) // 3 for n, op in re.findall(r"(\d+)([MID])", cigar or "") if op == "M")


def neural_v_classes(model, seqs, cfg, idx_class, device, batch=256):
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
            p = model({"tokenized_sequence": toks.to(device)})["v_allele"].cpu().numpy()
        out += [int(idx_class[int(p[j].argmax())]) for j in range(len(chunk))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/same_scheme/model.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=1500)
    ap.add_argument("--cap", type=int, default=800)          # max reads scored per cohort stratum
    ap.add_argument("--seed", type=int, default=29)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/same_scheme/derisk_aligner.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    model, ref, cfg, _ = setup_aa(args.dc, 192)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()
    vclass = build_class_maps(args.dc)[0]["V"].allele_class
    germ = {n: g for n, g in build_aa_germlines(getattr(__import__("GenAIRR"), args.dc),
                                                genes=("V",)).genes["V"].items()}
    classes, protos = v_class_prototypes(germ, vclass)
    idx_class = np.array([vclass.get(n, -1) for n in ref.gene("V").names])

    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    coh = defaultdict(list)                                  # stratum -> [{"aa","tc","corr"}]
    clean_pool = []
    for c in stream_benchmark(spec, dataconfig=None, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not r.get("sequence_aa") or r.get("junction_start") is None:
            continue
        call = r["v_call"].split(",")[0]
        if call not in vclass:
            continue
        rec = {"aa": r["sequence_aa"], "tc": vclass[call], "corr": _correctable_res(r.get("v_cigar", ""))}
        ni, nv = int(r.get("n_indels", 0)), int(r.get("n_v_indels", 0))
        if ni == 0 and c.stratum in ("clean_full", "moderate_full"):
            coh["clean"].append(rec)
            clean_pool.append(rec)
        elif nv == 1:
            coh["v_indel_1"].append(rec)
        elif nv >= 2:
            coh["v_indel_multi"].append(rec)
        elif ni > 0 and nv == 0:
            coh["dj_indel"].append(rec)
        if c.stratum == "amplicon_janchor":
            coh["short_janchor"].append(rec)

    # synthetic AA-native corruptions (the REAL protein-input error model), from clean reads
    for rec in clean_pool[:args.cap]:
        coh["aa_subs_7pct"].append({"aa": _aa_substitute(rec["aa"], 0.07, rng), "tc": rec["tc"], "corr": -1})
        coh["aa_single_indel"].append({"aa": _aa_indel(rec["aa"], rng), "tc": rec["tc"], "corr": -1})

    order = ["clean", "aa_subs_7pct", "aa_single_indel", "short_janchor",
             "dj_indel", "v_indel_1", "v_indel_multi"]
    results = {}
    for s in order:
        rows = coh.get(s, [])[:args.cap]
        if not rows:
            continue
        seqs = [r["aa"] for r in rows]
        tcs = np.array([r["tc"] for r in rows])
        neu = np.array(neural_v_classes(model, seqs, cfg, idx_class, args.device))
        aln = np.array([classify_v(s2, classes, protos)[0] for s2 in seqs])
        corr = [r["corr"] for r in rows if r["corr"] >= 0]
        results[s] = {"n": len(rows), "neural_acc": float((neu == tcs).mean()),
                      "aligner_acc": float((aln == tcs).mean()),
                      "diag_mean_correctable_res": (float(np.mean(corr)) if corr else None)}

    # predeclared rule
    vind = [r for s in ("v_indel_1", "v_indel_multi") for r in coh.get(s, [])[:args.cap]]
    if vind:
        seqs = [r["aa"] for r in vind]
        tcs = np.array([r["tc"] for r in vind])
        neu = np.array(neural_v_classes(model, seqs, cfg, idx_class, args.device))
        aln = np.array([classify_v(s2, classes, protos)[0] for s2 in seqs])
        vind_neu, vind_aln = float((neu == tcs).mean()), float((aln == tcs).mean())
    else:
        vind_neu = vind_aln = 0.0
    cl = results.get("clean", {})
    rule = {"vindel_neural": vind_neu, "vindel_aligner": vind_aln,
            "vindel_gain": vind_aln - vind_neu,
            "clean_neural": cl.get("neural_acc", 0.0), "clean_aligner": cl.get("aligner_acc", 0.0),
            "clean_loss": cl.get("neural_acc", 0.0) - cl.get("aligner_acc", 0.0)}
    rule["PROCEED"] = bool(rule["vindel_gain"] >= 0.05 and rule["clean_loss"] <= 0.02)

    report = {"per_stratum": results, "predeclared_rule": rule}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n==== DE-RISK: no-training local aligner vs neural head (V class acc) ====")
    print(f"{'stratum':16s} {'n':>5} {'neural':>7} {'aligner':>7} {'delta':>7}  (diag corr-res)")
    for s in order:
        if s not in results:
            continue
        d = results[s]
        dr = f"{d['diag_mean_correctable_res']:.0f}" if d["diag_mean_correctable_res"] is not None else "-"
        print(f"{s:16s} {d['n']:5d} {d['neural_acc']:7.3f} {d['aligner_acc']:7.3f} "
              f"{d['aligner_acc'] - d['neural_acc']:+7.3f}  ({dr})")
    print(f"\nPREDECLARED RULE:")
    print(f"  V-indel: neural {rule['vindel_neural']:.3f} -> aligner {rule['vindel_aligner']:.3f} "
          f"(gain {rule['vindel_gain']:+.3f}; need >= +0.05)")
    print(f"  clean:   neural {rule['clean_neural']:.3f} -> aligner {rule['clean_aligner']:.3f} "
          f"(loss {rule['clean_loss']:+.3f}; need <= +0.02)")
    print(f"  => {'PROCEED with re-ranker' if rule['PROCEED'] else 'DO NOT add re-ranker'}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
