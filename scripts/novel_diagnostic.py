"""Decompose the novel-allele recall gap (Property 1): is the 0.33 a RETRIEVAL problem
(cosine drops the novel germline from top-k) or a DISCRIMINATION/metric problem (novel is
retrieved but a real sibling out-ranks it — partly because an N-SNP synthetic novel is
genuinely farther from the read than a 1-SNP real sibling)?

For victim reads (true V allele A swapped for an unseen A~novel that is `snps` SNPs from A),
we report, as a function of perturbation distance:
  cosine_recall@k : A~novel is within the cosine top-k  (retrieval recall ceiling)
  reader_pick|in  : reader picks A~novel GIVEN it is in top-k (reader discrimination)
  final_call      : reader's top-1 == A~novel  (the end metric)
  miss=sibling    : when wrong, the pick is a real sibling of A (ambiguity, not failure)
"""
import argparse
import os
import random
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402
from heldout_alleles import build_novel_genotype  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


@torch.no_grad()
def diagnose(model, rs, recs, victims, rename, device, topks=(16, 32, 64)):
    reads = [r["sequence"] for r in recs]
    novel_genes, _ = None, None
    # build a novel reference where each victim is replaced by its ~novel variant
    # (rename already maps victim -> novel name; rebuild reference from the genes dict)
    # predict with scores emitted so we can read the rank of the novel allele
    preds = predict_reads(model, rs, reads, device=device, topk=max(topks),
                          rerank="learned", emit_scores=True)
    vnames = rs.gene("V").names
    idx = {n: i for i, n in enumerate(vnames)}
    stats = {k: [0, 0] for k in topks}              # recall@k: [hits, total]
    pick_given_in = [0, 0]
    final = [0, 0]
    miss_sibling = 0
    for rec, p in zip(recs, preds):
        v = str(rec.get("v_call", "")).split(",")[0]
        if v not in victims:
            continue
        target = rename[v]                          # the ~novel name = correct answer
        scored = p.get("v_scores") or []
        order = [nm for nm, _ in sorted(scored, key=lambda x: -x[1])]   # reader-ranked
        topk_names = p.get("v_topk", [])            # cosine-retrieved candidates (<= max topk)
        for k in topks:
            stats[k][1] += 1
            stats[k][0] += int(target in set(topk_names[:k]))
        in_top = target in set(topk_names)
        if in_top:
            pick_given_in[1] += 1
            pick_given_in[0] += int(p["v_call"] == target)
        final[1] += 1
        final[0] += int(p["v_call"] == target)
        if p["v_call"] != target:
            # was the pick a real sibling (same gene family as A)?
            picked_gene = p["v_call"].split("*")[0]
            if picked_gene == v.split("*")[0]:
                miss_sibling += 1
    return stats, pick_given_in, final, miss_sibling


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_novel.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--n-victims", type=int, default=20)
    ap.add_argument("--p", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    full_rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    print(f"model={args.model}\n")

    recs = gen_records(args.p, args.n, args.seed, None)
    true_v = [str(r.get("v_call", "")).split(",")[0] for r in recs]
    present = [v for v in dict.fromkeys(true_v) if v]

    for snps in (1, 2, 3):
        rng = random.Random(args.seed)
        victims = set(rng.sample(present, min(args.n_victims, len(present))))
        genes, rename = build_novel_genotype(full_rs, victims, snps, rng)
        novel_rs = ReferenceSet.from_genotype(genes)
        stats, pgi, final, miss_sib = diagnose(model, novel_rs, recs, victims, rename, device)
        nvic = final[1]
        print(f"[snps={snps}]  victim reads={nvic}")
        for k in (16, 32, 64):
            print(f"    cosine_recall@{k:<3} = {stats[k][0] / max(stats[k][1], 1):.3f}")
        print(f"    reader_pick | in_topk = {pgi[0] / max(pgi[1], 1):.3f}  (discrimination)")
        print(f"    final_call            = {final[0] / max(final[1], 1):.3f}")
        print(f"    of misses, %sibling   = {miss_sib / max(final[1] - final[0], 1):.3f}\n")


if __name__ == "__main__":
    main()
