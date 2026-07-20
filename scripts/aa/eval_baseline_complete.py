"""Stage-0 baseline completion for the frozen strategic V/D/J model (no retraining).

Adds what §17's quick eval lacked, per review:
  - J measured on the cohort (model + alignment reference);
  - the local aligner is a "reference", NOT a ceiling (the model beats it on V-indels);
  - Wilson intervals on every accuracy; D reported micro AND macro (per-class-balanced);
  - a D-EVIDENCE ABLATION: mask the observed junction/D residues (V-end..J-start) with X and
    re-measure D — if D collapses toward the prior, the call comes from observable D residues, not
    from correlated rearrangement context;
  - the checkpoint SHA-256 recorded in the artifact.

Run: PYTHONPATH=src .venv/bin/python scripts/aa/eval_baseline_complete.py --device cuda
"""
import argparse
import hashlib
import json
import os
from collections import Counter, defaultdict

import numpy as np
import torch

import GenAIRR as ga
from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.data import build_class_maps
from alignair.aa.evaluate import wilson_interval
from alignair.aa.gym_bridge import aa_usable
from alignair.aa.local_aligner import classify_v, v_class_prototypes
from alignair.aa.reference import build_aa_germlines
from alignair.aa.strategic import strategic_setup
from alignair.aa.tokenizer import AAProteinTokenizer
from alignair.aa.translation import frame_offset_of, project_codons
from alignair.reference.reference_set import ReferenceSet


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _predict(model, seqs, cfg, device, key, batch=256):
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
            lg = model({"tokenized_sequence": toks.to(device)})[key].cpu().numpy()
        out += [int(lg[j].argmax()) for j in range(len(chunk))]
    return out


def _acc_ci(correct, n):
    p = correct / n if n else 0.0
    lo, hi = wilson_interval(correct, n)
    return {"acc": p, "n": n, "ci95": [lo, hi]}


def _macro(true, pred):
    per = defaultdict(lambda: [0, 0])
    for t, p in zip(true, pred):
        per[t][1] += 1
        per[t][0] += int(t == p)
    return float(np.mean([h / n for h, n in per.values()])) if per else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/strategic/model_vdj_40k.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=1500)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=41)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="docs/superpowers/artifacts/2026-07-20-aa-strategic/baseline_complete.json")
    args = ap.parse_args()

    model, cfg, class_maps, meta = strategic_setup(args.dc, 192, has_d=True)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()
    sha = _sha256(args.model)
    dc = getattr(ga, args.dc)
    ref = ReferenceSet.from_dataconfigs(dc)
    vclass, jclass = class_maps["V"].allele_class, class_maps["J"].allele_class
    d_index = meta["d_index"]
    d_name = {i: n for n, i in d_index.items()}
    gsets = build_aa_germlines(dc, genes=("V", "J"))
    v_classes, v_protos = v_class_prototypes(gsets.genes["V"], vclass)
    j_classes, j_protos = v_class_prototypes(gsets.genes["J"], jclass)

    # cohort: per read keep aa, v/j class, d call, and the junction span [v_end_aa, j_start_aa)
    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    coh = defaultdict(list)
    for c in stream_benchmark(spec, dataconfig=dc, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not aa_usable(r):
            continue
        vc = r["v_call"].split(",")[0]
        jc = r["j_call"].split(",")[0]
        if vc not in vclass or jc not in jclass:
            continue
        segs = {}
        for g, G in (("v", "V"), ("d", "D"), ("j", "J")):
            s, e = r.get(f"{g}_sequence_start"), r.get(f"{g}_sequence_end")
            if s is not None and e is not None:
                segs[G] = (int(s), int(e))
        proj = project_codons(len(r["sequence"]), frame_offset_of(r["junction_start"]), segs)
        vb, jb = proj.aa_bounds.get("V"), proj.aa_bounds.get("J")
        if vb is None or jb is None:
            continue
        dcall = (r.get("d_call") or "").split(",")[0].strip()
        rec = {"aa": r["sequence_aa"], "vc": vclass[vc], "jc": jclass[jc],
               "dcall": dcall, "dinv": bool(r.get("d_inverted", False)),
               "vend": vb[1], "jstart": jb[0]}          # junction span for the D-evidence ablation
        ni, nv = int(r.get("n_indels", 0)), int(r.get("n_v_indels", 0))
        key = ("clean" if ni == 0 and c.stratum in ("clean_full", "moderate_full")
               else "v_indel" if nv > 0 else "dj_indel" if ni > 0 else None)
        if key:
            coh[key].append(rec)

    report = {"checkpoint": args.model, "sha256": sha, "n_v_classes": meta["n_v_classes"],
              "n_d_alleles": meta["n_d_alleles"], "strata": {}}

    for s in ("clean", "v_indel", "dj_indel"):
        rows = coh.get(s, [])[:args.cap]
        if not rows:
            continue
        seqs = [r["aa"] for r in rows]
        # V + J: model vs alignment REFERENCE (not a ceiling)
        vt = [r["vc"] for r in rows]
        jt = [r["jc"] for r in rows]
        vp = _predict(model, seqs, cfg, args.device, "v_class_logits")
        jp = _predict(model, seqs, cfg, args.device, "j_class_logits")
        vp_ref = [classify_v(x, v_classes, v_protos)[0] for x in seqs]
        jp_ref = [classify_v(x, j_classes, j_protos)[0] for x in seqs]
        out = {"n": len(rows),
               "V_model": _acc_ci(sum(a == b for a, b in zip(vp, vt)), len(vt)),
               "V_align_ref": _acc_ci(sum(a == b for a, b in zip(vp_ref, vt)), len(vt)),
               "J_model": _acc_ci(sum(a == b for a, b in zip(jp, jt)), len(jt)),
               "J_align_ref": _acc_ci(sum(a == b for a, b in zip(jp_ref, jt)), len(jt))}
        # D: micro + macro (allele + gene) vs frequency prior, on valid-D reads
        d_rows = [(i, r) for i, r in enumerate(rows) if r["dcall"] in d_index and not r["dinv"]]
        if d_rows:
            idx = [i for i, _ in d_rows]
            dt = [rows[i]["dcall"] for i in idx]
            dp = [d_name[k] for k in [_predict(model, [seqs[i] for i in idx], cfg, args.device,
                                                "d_class_logits")][0]]
            gene = lambda n: n.split("*")[0]
            nD = len(dt)
            out["D"] = {
                "n": nD,
                "allele_micro": _acc_ci(sum(a == b for a, b in zip(dp, dt)), nD),
                "gene_micro": _acc_ci(sum(gene(a) == gene(b) for a, b in zip(dp, dt)), nD),
                "allele_macro": _macro(dt, dp),
                "allele_prior": Counter(dt).most_common(1)[0][1] / nD,
                "gene_prior": Counter(gene(t) for t in dt).most_common(1)[0][1] / nD,
            }
            # D-EVIDENCE ABLATION: mask the observed junction/D residues (V-end..J-start) with X
            masked = []
            for i in idx:
                aa = rows[i]["aa"]
                a, b = rows[i]["vend"], rows[i]["jstart"]
                masked.append(aa[:a] + "X" * max(0, b - a) + aa[b:])
            dp_mask = [d_name[k] for k in _predict(model, masked, cfg, args.device, "d_class_logits")]
            out["D"]["allele_micro_junction_masked"] = _acc_ci(
                sum(a == b for a, b in zip(dp_mask, dt)), nD)
        report["strata"][s] = out

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"\n==== Stage-0 baseline completion ({args.model}) sha {sha[:12]}… ====")
    for s, o in report["strata"].items():
        print(f"\n[{s}] n={o['n']}")
        for g in ("V", "J"):
            m, rf = o[f"{g}_model"], o[f"{g}_align_ref"]
            print(f"  {g}: model {m['acc']:.3f} CI[{m['ci95'][0]:.3f},{m['ci95'][1]:.3f}]  "
                  f"align-ref {rf['acc']:.3f}")
        if "D" in o:
            d = o["D"]
            print(f"  D allele micro {d['allele_micro']['acc']:.3f} "
                  f"CI[{d['allele_micro']['ci95'][0]:.3f},{d['allele_micro']['ci95'][1]:.3f}]  "
                  f"macro {d['allele_macro']:.3f}  (prior {d['allele_prior']:.3f})  "
                  f"gene {d['gene_micro']['acc']:.3f}")
            print(f"  D-evidence ablation (junction masked): allele "
                  f"{d['allele_micro_junction_masked']['acc']:.3f}  "
                  f"(drop {d['allele_micro']['acc'] - d['allele_micro_junction_masked']['acc']:+.3f})")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
