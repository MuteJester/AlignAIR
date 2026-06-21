"""Falsification experiment (NO retraining): is candidate-generation capped by pooled
cosine, and does late-interaction (MaxSim) + exact-kmer seeding over the DEEP backbone
reps break the recall ceiling (topk_truth_recall 0.814, oracle-cosine top-1 0.365)?

For an existing --backbone shared checkpoint we score the SAME oracle-segment reads four
ways and report topk_truth_recall@K + top-1 call accuracy per gene/difficulty:
  cosine   : the current shallow-GermlineEncoder pooled cosine (out["match"])
  maxsim   : ColBERT late-interaction over backbone per-position reps (mean_i max_j cos)
  kmer     : exact k-mer seed overlap (BLAST-style, no params) over raw tokens
  maxsim+kmer : the proposed combined recall score

If maxsim/maxsim+kmer lift topk recall from ~0.81 past ~0.95 on heavy-SHM with NO
training, the diagnosis is confirmed and the late-interaction head is worth building.
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(__file__))

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR, extract_segment, extract_segment_tokens  # noqa: E402
from alignair.data.tokenizer import pad_tokenize  # noqa: E402
from alignair.gym.gym import AlignAIRGym  # noqa: E402
from alignair.gym.curriculum import Curriculum  # noqa: E402
from alignair.training.gym_trainer import GymTrainer  # noqa: E402

GERMLINE = 1


def maxsim_score(seg, seg_mask, germ, germ_mask, kchunk=8):
    """Late-interaction MaxSim: mean over read positions of the max cosine to any germline
    position. seg (B,S,d) and germ (K,Lg,d) are L2-normalized. Returns (B,K)."""
    B, S, d = seg.shape
    K = germ.shape[0]
    out = seg.new_zeros(B, K)
    denom = seg_mask.sum(dim=1).clamp(min=1).unsqueeze(1).to(seg.dtype)     # (B,1)
    for s in range(0, K, kchunk):
        gk, gm = germ[s:s + kchunk], germ_mask[s:s + kchunk]               # (c,Lg,d),(c,Lg)
        sim = torch.einsum("bsd,cld->bcsl", seg, gk)                        # (B,c,S,Lg)
        sim = sim.masked_fill(~gm[None, :, None, :], float("-inf"))
        mx = sim.max(dim=-1).values                                        # (B,c,S)
        mx = mx.masked_fill(~seg_mask[:, None, :], 0.0)
        out[:, s:s + kchunk] = mx.sum(dim=-1) / denom
    return out


def kmer_counts(tok, mask, k=5, V=4):
    """(N, V**k) length-normalizable counts of exact k-mers (tokens 1..4); invalid/pad
    k-mers are routed to a dump bucket and dropped."""
    N, L = tok.shape
    D = V ** k
    nkm = L - k + 1
    if nkm <= 0:
        return tok.new_zeros(N, D, dtype=torch.float)
    base = (tok - 1).clamp(0, V - 1)
    valid = mask & (tok >= 1) & (tok <= 4)
    idx = tok.new_zeros(N, nkm, dtype=torch.long)
    valk = torch.ones(N, nkm, dtype=torch.bool, device=tok.device)
    for j in range(k):
        idx = idx + base[:, j:j + nkm] * (V ** j)
        valk = valk & valid[:, j:j + nkm]
    idx = idx.masked_fill(~valk, D)                                        # dump bucket
    counts = tok.new_zeros(N, D + 1, dtype=torch.float)
    counts.scatter_add_(1, idx, valk.float())
    return counts[:, :D]


def kmer_score(seg_tok, seg_mask, germ_tok, germ_mask, k=5):
    sc = kmer_counts(seg_tok, seg_mask, k)
    gc = kmer_counts(germ_tok, germ_mask, k)
    return (sc @ gc.t()) / sc.sum(dim=1, keepdim=True).clamp(min=1)        # (B,K) seed overlap


def recall_top1(score, multihot, k=32):
    topk = score.topk(min(k, score.shape[1]), dim=-1).indices
    in_topk = (multihot.gather(1, topk).sum(dim=1) > 0).float().mean().item()
    top1 = (multihot.gather(1, score.argmax(-1, keepdim=True)).squeeze(1) > 0).float().mean().item()
    return in_topk, top1


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--topk", type=int, default=32)
    ap.add_argument("--kmer", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    assert getattr(cfg, "backbone", "conv") == "shared", "need a --backbone shared checkpoint"
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"]); model.eval()
    print(f"loaded {args.model}  (backbone={cfg.backbone})  topk={args.topk}  kmer={args.kmer}")

    ref_emb = model.encode_reference(rs)                                   # cosine path
    genes = ["v", "j"] + (["d"] if rs.has_d else [])
    # germline per-position reps from the DEEP backbone (token_type=GERMLINE) + raw tokens
    gb = {}
    for g in genes:
        G = g.upper()
        tok, msk = pad_tokenize(rs.gene(G).sequences)
        tok, msk = tok.to(device), msk.to(device)
        reps = F.normalize(model.backbone.forward_positions(tok, msk, GERMLINE), dim=-1)
        gb[G] = {"reps": reps, "mask": msk, "tok": tok}

    gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, seed=0, curriculum=Curriculum())
    trainer = GymTrainer(model, torch.nn.Identity(), rs, gym, batch_size=args.batch, device=device)
    scorers = ["cosine", "maxsim", "kmer", "maxsim+kmer"]

    for label, p in [("clean", 0.0), ("hard/heavy-SHM", 1.0)]:
        gym.set_progress(p)
        loader = trainer._loader()
        agg = {g: {sc: [0.0, 0.0, 0] for sc in scorers} for g in genes}
        for nb, batch in enumerate(loader):
            if nb >= args.batches:
                break
            batch = trainer._to_device(batch)
            out = model(batch["tokens"], batch["mask"], ref_emb,
                        orientation_ids=batch["orientation_id"])
            reps, canon = out["reps"], out["canon_tokens"]
            true_region = batch["region_labels"]                          # ORACLE segments
            for g in genes:
                G = g.upper()
                seg_reps, seg_mask = extract_segment(reps, batch["mask"], true_region, G)
                seg_reps = F.normalize(seg_reps, dim=-1)
                seg_tok, _ = extract_segment_tokens(canon, batch["mask"], true_region, G)
                mh = batch[f"{g}_allele"]
                ms = maxsim_score(seg_reps, seg_mask, gb[G]["reps"], gb[G]["mask"])
                km = kmer_score(seg_tok, seg_mask, gb[G]["tok"], gb[G]["mask"], args.kmer)
                S = {"cosine": out["match"][G], "maxsim": ms, "kmer": km,
                     "maxsim+kmer": ms + km}
                for sc in scorers:
                    r, t = recall_top1(S[sc], mh, args.topk)
                    agg[g][sc][0] += r; agg[g][sc][1] += t; agg[g][sc][2] += 1
        print(f"\n[{label}]   (oracle segments; recall@{args.topk} / top1)")
        for g in genes:
            print(f"  {g.upper()}:")
            for sc in scorers:
                r, t, n = agg[g][sc]
                print(f"     {sc:12s} recall={r / n:.3f}  top1={t / n:.3f}")


if __name__ == "__main__":
    main()
