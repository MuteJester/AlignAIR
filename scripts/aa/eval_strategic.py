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


def strategic_pred(model, seqs, cfg, device, key, batch=256):
    """argmax of a class-logit head (`v_class_logits` or `d_class_logits`) over a list of AA strings."""
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


def strategic_v_classes(model, seqs, cfg, device, batch=256):
    return strategic_pred(model, seqs, cfg, device, "v_class_logits", batch)


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
        dcall = (r.get("d_call") or "").split(",")[0].strip()
        rec = {"aa": r["sequence_aa"], "tc": vclass[call],
               "d_call": dcall, "d_inv": bool(r.get("d_inverted", False))}
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

    def _syn(aa, rec):                                        # keep V/D truth, swap the AA sequence
        return {"aa": aa, "tc": rec["tc"], "d_call": rec["d_call"], "d_inv": rec["d_inv"]}
    for rec in clean_pool[:args.cap]:
        coh["aa_subs_7pct"].append(_syn(aa_substitute(rec["aa"], 0.07, rng), rec))
        coh["aa_single_indel"].append(_syn(aa_indel(rec["aa"], rng), rec))
        coh["aa_il_ambiguity"].append(_syn(il_ambiguity(rec["aa"]), rec))
        coh["aa_truncate_50_80"].append(_syn(aa_truncate(rec["aa"], 0.5, 0.8, rng), rec))

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

    # ---- D allele evaluation (honest: allele + gene level vs the D-frequency prior) ----
    d_index = meta["d_index"]
    d_name = {i: n for n, i in d_index.items()}
    def d_gene(name):
        return name.split("*")[0]
    d_report = {}
    for s in ("clean", "aa_subs_7pct", "aa_single_indel"):
        rows = [r for r in coh.get(s, [])[:args.cap] if r["d_call"] in d_index and not r["d_inv"]]
        if not rows:
            continue
        seqs = [r["aa"] for r in rows]
        true = [r["d_call"] for r in rows]
        pred = [d_name[i] for i in strategic_pred(model, seqs, cfg, args.device, "d_class_logits")]
        allele_acc = float(np.mean([p == t for p, t in zip(pred, true)]))
        gene_acc = float(np.mean([d_gene(p) == d_gene(t) for p, t in zip(pred, true)]))
        from collections import Counter
        maj = Counter(true).most_common(1)[0][1] / len(true)
        maj_gene = Counter(d_gene(t) for t in true).most_common(1)[0][1] / len(true)
        d_report[s] = {"n": len(rows), "d_allele_acc": allele_acc, "d_gene_acc": gene_acc,
                       "d_allele_prior": maj, "d_gene_prior": maj_gene}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"per_stratum": results, "d_eval": d_report, "model": args.model}, f,
                  indent=2, sort_keys=True)

    print(f"\n==== STRATEGIC AA model (class heads) vs alignment reference — V class acc ====")
    print(f"{'stratum':18s} {'n':>5} {'model':>7} {'aligner':>7} {'gap':>7}")
    for s in order:
        if s not in results:
            continue
        d = results[s]
        print(f"{s:18s} {d['n']:5d} {d['model_acc']:7.3f} {d['aligner_acc']:7.3f} "
              f"{d['model_acc'] - d['aligner_acc']:+7.3f}")
    print(f"\n==== D allele head (honest: vs D-frequency prior) ====")
    print(f"{'stratum':18s} {'n':>5} {'D-allele':>9} {'(prior)':>8} {'D-gene':>8} {'(prior)':>8}")
    for s, d in d_report.items():
        print(f"{s:18s} {d['n']:5d} {d['d_allele_acc']:9.3f} {d['d_allele_prior']:8.3f} "
              f"{d['d_gene_acc']:8.3f} {d['d_gene_prior']:8.3f}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
