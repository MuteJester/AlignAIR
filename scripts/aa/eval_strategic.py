"""Evaluate the strategic AA model (protein-equivalence CLASS heads) against the no-training
alignment reference across AA-native strata (the real protein-input error model) + indel strata.

Answers: did explicit class targets + AA-native augmentation + longer training close the gap to the
alignment ceiling (aligner ~0.99 clean / ~0.94 AA-native)?

Run: PYTHONPATH=src .venv/bin/python scripts/aa/eval_strategic.py --model experiments/aa/strategic/model.pt --device cuda
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np
import torch

from alignair_benchmark import default_igh_spec, stream_benchmark

from alignair.aa.augment import aa_indel, aa_substitute, aa_truncate, il_ambiguity
from alignair.aa.data import build_class_maps
from alignair.aa.local_aligner import classify_v, v_class_prototypes
from alignair.aa.reference import build_aa_germlines
from alignair.aa.strategic import strategic_setup
from alignair.aa.tokenizer import AAProteinTokenizer


def strategic_v_classes(model, seqs, cfg, device, batch=256):
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
            lg = model({"tokenized_sequence": toks.to(device)})["v_class_logits"].cpu().numpy()
        out += [int(lg[j].argmax()) for j in range(len(chunk))]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="experiments/aa/strategic/model.pt")
    ap.add_argument("--dc", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n-per-stratum", type=int, default=1200)
    ap.add_argument("--cap", type=int, default=800)
    ap.add_argument("--seed", type=int, default=31)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--out", default="experiments/aa/strategic/eval.json")
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)

    import GenAIRR as ga
    from alignair.reference.reference_set import ReferenceSet

    model, cfg, class_maps, meta = strategic_setup(args.dc, 192)
    model.load_state_dict(torch.load(args.model, map_location="cpu")["model"])
    model.to(args.device).eval()
    dc = getattr(ga, args.dc)
    ref = ReferenceSet.from_dataconfigs(dc)
    vclass = class_maps["V"].allele_class
    germ = {n: g for n, g in build_aa_germlines(dc, genes=("V",)).genes["V"].items()}
    classes, protos = v_class_prototypes(germ, vclass)

    spec = default_igh_spec(n_per_stratum=args.n_per_stratum, seed=args.seed)
    coh = defaultdict(list)
    clean_pool = []
    for c in stream_benchmark(spec, dataconfig=dc, reference_set=ref):
        r = c.record
        if c.orientation_id != 0 or not r.get("sequence_aa") or r.get("junction_start") is None:
            continue
        call = r["v_call"].split(",")[0]
        if call not in vclass:
            continue
        rec = {"aa": r["sequence_aa"], "tc": vclass[call]}
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

    for rec in clean_pool[:args.cap]:
        coh["aa_subs_7pct"].append({"aa": aa_substitute(rec["aa"], 0.07, rng), "tc": rec["tc"]})
        coh["aa_single_indel"].append({"aa": aa_indel(rec["aa"], rng), "tc": rec["tc"]})
        coh["aa_il_ambiguity"].append({"aa": il_ambiguity(rec["aa"]), "tc": rec["tc"]})
        coh["aa_truncate_50_80"].append({"aa": aa_truncate(rec["aa"], 0.5, 0.8, rng), "tc": rec["tc"]})

    order = ["clean", "aa_subs_7pct", "aa_single_indel", "aa_il_ambiguity", "aa_truncate_50_80",
             "short_janchor", "dj_indel", "v_indel_1", "v_indel_multi"]
    results = {}
    for s in order:
        rows = coh.get(s, [])[:args.cap]
        if not rows:
            continue
        seqs = [r["aa"] for r in rows]
        tcs = np.array([r["tc"] for r in rows])
        neu = np.array(strategic_v_classes(model, seqs, cfg, args.device))
        aln = np.array([classify_v(s2, classes, protos)[0] for s2 in seqs])
        results[s] = {"n": len(rows), "model_acc": float((neu == tcs).mean()),
                      "aligner_acc": float((aln == tcs).mean())}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_stratum": results, "model": args.model}, f, indent=2, sort_keys=True)

    print(f"\n==== STRATEGIC AA model (class heads) vs alignment reference — V class acc ====")
    print(f"{'stratum':18s} {'n':>5} {'model':>7} {'aligner':>7} {'gap':>7}")
    for s in order:
        if s not in results:
            continue
        d = results[s]
        print(f"{s:18s} {d['n']:5d} {d['model_acc']:7.3f} {d['aligner_acc']:7.3f} "
              f"{d['model_acc'] - d['aligner_acc']:+7.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
