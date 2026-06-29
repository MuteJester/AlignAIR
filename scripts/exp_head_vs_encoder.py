"""Head-vs-encoder probe: is the heavy-SHM V weakness fixable with a better ALLELE HEAD on the
frozen encoder, or is the ENCODER itself the wall?

Freezes the trained encoder and trains ONLY a small learned cross-attention allele head
(segment per-position reps = queries, candidate germline per-position reps = keys/values) with
set-NCE. Then evaluates the head AS A CALLER (scores ALL alleles, argmax) per cell, against:
  pooled   : the model's pooled-cosine retrieval argmax (current caller)
  xattn    : the trained cross-attention head argmax over ALL alleles
and reports recall@16 for pooled vs xattn (does cross-attention raise the retrieval ceiling?).

Verdict:
  xattn >> pooled  -> the per-position reps DO carry allele signal pooling throws away; a learned
                      head is the fix (and it's one attention op, cheap).
  xattn ~ pooled   -> the frozen reps lack the separating signal -> the ENCODER is the wall ->
                      prioritize a pretrained/stronger encoder over head surgery.

Run:
  PYTHONPATH=src .venv/bin/python scripts/exp_head_vs_encoder.py \
      --model .private/models/seed_extend_d64_reader.pt --steps 600 --eval-n 200
"""
import argparse
import random

import torch
import torch.nn as nn
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.stats import bootstrap_ci
from alignair.training.reader import build_sibling_index, build_candidates, reader_set_nce
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


class XAttnAlleleHead(nn.Module):
    """Multi-head cross-attention soft-aligner: a trainable generalization of MaxSim. Segment
    positions attend over a candidate allele's germline positions; a learned per-position
    compatibility readout is masked-averaged to one score per (read, candidate)."""
    def __init__(self, d: int, heads: int = 4):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v = nn.Linear(d, d), nn.Linear(d, d), nn.Linear(d, d)
        self.out = nn.Linear(d, d)
        self.readout = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))

    def forward(self, seg, seg_mask, germ, germ_mask):
        N, S, d = seg.shape
        Lg = germ.shape[1]
        Q = self.q(seg).view(N, S, self.h, self.dh).transpose(1, 2)        # (N,h,S,dh)
        K = self.k(germ).view(N, Lg, self.h, self.dh).transpose(1, 2)
        V = self.v(germ).view(N, Lg, self.h, self.dh).transpose(1, 2)
        att = (Q @ K.transpose(-1, -2)) / (self.dh ** 0.5)                 # (N,h,S,Lg)
        att = att.masked_fill(~germ_mask[:, None, None, :], -1e9)
        ctx = (torch.softmax(att, dim=-1) @ V).transpose(1, 2).reshape(N, S, d)
        pos = self.readout(self.out(ctx)).squeeze(-1)                      # (N,S)
        pos = pos.masked_fill(~seg_mask, 0.0)
        return pos.sum(-1) / seg_mask.sum(-1).clamp(min=1)                 # (N,)


def _v_segment(model, batch, ref_emb, device):
    out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
    pr = out["region_logits"].argmax(-1)
    seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"], pr, "V")
    seg_reps, _ = extract_segment(out["reps"], batch["mask"], pr, "V")
    return out, seg_reps, seg_tok, seg_mask


def _score_pairs(head, seg_reps, seg_mask, read_ix, cand_ix, pos_reps, pos_mask, chunk=2048):
    S, d = seg_reps.shape[1], seg_reps.shape[2]
    parts = []
    for a in range(0, read_ix.shape[0], chunk):
        ri, ci = read_ix[a:a + chunk], cand_ix[a:a + chunk]
        parts.append(head(seg_reps[ri], seg_mask[ri], pos_reps[ci], pos_mask[ci]))
    return torch.cat(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/seed_extend_d64_reader.pt")
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--eval-n", type=int, default=200)
    ap.add_argument("--k", type=int, default=16)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                                # FREEZE encoder
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    pos_reps, pos_mask = ref_emb["V"]["pos_reps"], ref_emb["V"]["pos_mask"]
    A = pos_reps.shape[0]
    d = ck["config"]["d_model"]
    head = XAttnAlleleHead(d).to(device)
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    sib = build_sibling_index(rs)["V"]
    rng = random.Random(0)

    # ---- train the head only ----
    gym = AlignAIRGym([dc], rs, n=a.batch_size * 8, seed=0, curriculum=StratifiedCurriculum())
    loader = DataLoader(gym, batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
    head.train(); step = 0; running = 0.0
    it = iter(loader)
    while step < a.steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader); batch = next(it)
        batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
        with torch.no_grad():
            _, seg_reps, seg_tok, seg_mask = _v_segment(model, batch, ref_emb, device)
        prim = batch["v_primary_idx"]; mh = batch["v_allele"]
        cand, pos_m = build_candidates(prim, mh, sib, rng, n_sib=8, n_rand=8)   # (B,C)
        B, C = cand.shape
        read_ix = torch.arange(B, device=device).repeat_interleave(C)
        sc = _score_pairs(head, seg_reps, seg_mask, read_ix, cand.reshape(-1),
                          pos_reps, pos_mask).reshape(B, C)
        loss = reader_set_nce(sc, pos_m)
        opt.zero_grad(); loss.backward(); opt.step()
        running += float(loss); step += 1
        if step % 150 == 0:
            print(f"  step {step}/{a.steps}  set-NCE {running / 150:.4f}", flush=True)
            running = 0.0

    # ---- evaluate head AS A CALLER (scores ALL alleles) vs pooled retrieval ----
    head.eval()
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}
    print(f"\n{'cell':18s} {'pooled_acc':>11s} {'xattn_acc':>10s} | {'pooled@'+str(a.k):>9s} {'xattn@'+str(a.k):>9s}")
    for cname in CELLS:
        cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                             "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
        el = DataLoader(AlignAIRGym([dc], rs, n=a.eval_n, seed=0, curriculum=cur),
                        batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        pa, xa, pr_, xr_ = [], [], [], []
        with torch.no_grad():
            for batch in el:
                batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
                out, seg_reps, seg_tok, seg_mask = _v_segment(model, batch, ref_emb, device)
                B = seg_reps.shape[0]; mh = batch["v_allele"]
                # head scores over ALL alleles
                read_ix = torch.arange(B, device=device).repeat_interleave(A)
                cand_ix = torch.arange(A, device=device).repeat(B)
                sc = _score_pairs(head, seg_reps, seg_mask, read_ix, cand_ix,
                                  pos_reps, pos_mask).reshape(B, A)
                x_top1 = sc.argmax(1); x_topk = sc.topk(min(a.k, A), dim=1).indices
                p_top1 = out["match"]["V"].argmax(1); p_topk = out["match"]["V"].topk(min(a.k, A), dim=1).indices
                pa.append(float((mh.gather(1, p_top1[:, None]).squeeze(1) > 0).float().mean()))
                xa.append(float((mh.gather(1, x_top1[:, None]).squeeze(1) > 0).float().mean()))
                pr_.append(float((mh.gather(1, p_topk) > 0).any(1).float().mean()))
                xr_.append(float((mh.gather(1, x_topk) > 0).any(1).float().mean()))
        def m(xs):
            return bootstrap_ci(xs)[0] if xs else float("nan")
        print(f"{cname:18s} {m(pa):11.3f} {m(xa):10.3f} | {m(pr_):9.3f} {m(xr_):9.3f}")
    print("\nxattn_acc >> pooled_acc and xattn@k > pooled@k  => head is the fixable lever (cheap).")
    print("xattn ~ pooled                                  => frozen encoder is the wall (pretrain).")


if __name__ == "__main__":
    main()
