"""Current-encoder Gate-2 DRY RUN (Codex's recommended next move): on the FROZEN 8h model,
test the integrated seed-and-extend pipeline under PREDICTED regions + top-k retrieval (Gate 1
was true-region/true-allele only). Cheapest go/no-go on the multiplicative failure mode
  deploy_quality ~ retrieval_recall@k  x  band_recall | (true allele in top-k)  x  fail-open.
No retrain (the band head is trained Gate-1-style on the mixture; the DP params are copied from
the trained soft-DP). Per lattice cell reports:
  - retr_recall@k     : is the true V allele in the retrieval top-k (predicted region segment)?
  - band_rec|topk     : band head (on the PREDICTED segment) covers true germline_start for the
                        true allele, among reads where the true allele is in top-k (top-m union).
  - retr_top1_acc     : model's own retrieval argmax allele correct (baseline).
  - dp_rerank_acc     : DP log-partition rerank over top-k picks a true allele (rule-1 reader).
  - fail_open / budget: band-head peak_evidence fail-open on the true-allele band (speed proxy).

STOP if retr_recall@16 is weak, band_rec|topk drops materially below Gate 1 (~1.0 clean / 0.97
junction), dp_rerank_acc < retr_top1_acc, or fail-open/budget erases the speed win.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_seed_extend_gate2_dryrun.py --train-steps 2000 --n 150 --k 16
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens, extract_segment
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.stats import bootstrap_ci
from alignair.gym.instrument.band_metrics import _topm_centers
from alignair.nn.aligner.band_head import BandHead, band_offset_loss, peak_evidence
from alignair.nn.aligner.banded_dp import SeedExtendAligner
from alignair.nn.heads.state import state_reliability
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def _loader(dc, rs, curriculum, n, bs, seed):
    gym = AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=curriculum)
    return DataLoader(gym, batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def _cell_curr(lat, cell):
    return type("C", (), {"params": lambda s, p=0.0: dict(lat.cell_params(cell)),
                          "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--topm", type=int, default=2)
    ap.add_argument("--ev-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}
    pos_reps, pos_mask, pos_tok = ref_emb["V"]["pos_reps"], ref_emb["V"]["pos_mask"], ref_emb["V"]["pos_tok"]

    # DP reader: SeedExtendAligner with params copied from the trained soft-DP
    sd = model.aligner
    al = SeedExtendAligner(d_model=ck["config"]["d_model"]).to(device).eval()
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        for p in ("log_scale", "_gap_open", "_gap_extend", "_del_gap", "_match_weight"):
            getattr(al, p).copy_(getattr(sd, p))

    # train the band head Gate-1 style (true region/allele, mixture stream)
    head = BandHead(d_model=ck["config"]["d_model"]).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)
    tr = _loader(dc, rs, StratifiedCurriculum(), a.n * 60, a.batch_size, seed=1)
    it = iter(tr); step = 0
    while step < a.train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(tr); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
            seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"],
                                                       batch["region_labels"], "V")
            seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
            idx = batch["v_primary_idx"]
        logits = head(seg_reps, seg_mask, pos_reps[idx], pos_mask[idx], seg_tok, pos_tok[idx])
        loss = band_offset_loss(logits, batch["v_germline_start"])
        opt.zero_grad(); loss.backward(); opt.step(); step += 1
        if step % 500 == 0:
            print(f"[band-train] {step}/{a.train_steps} loss {float(loss):.3f}", flush=True)
    head.eval()

    print(f"\nGate-2 DRY RUN (frozen model, PREDICTED region + top-{a.k}) | w={a.w} top-m={a.topm}")
    print(f"{'cell':18s} {'retr@k':>10s} {'band|topk':>11s} {'retr_top1':>10s} {'dp_rerank':>10s} "
          f"{'failopen':>9s} {'budget':>9s}")
    Kv = pos_reps.shape[0]
    for cname in CELLS:
        loader = _loader(dc, rs, _cell_curr(lat, cells[cname]), a.n, a.batch_size, seed=0)
        rr, br, t1, dp, fo, cb = [], [], [], [], [], []
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                pred_region = out["region_logits"].argmax(-1)
                seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"], pred_region, "V")
                seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
                B, S = seg_tok.shape
                topk = out["match"]["V"].topk(min(a.k, Kv), dim=-1).indices         # (B,k)
                k = topk.shape[1]
                prim = batch["v_primary_idx"]; multihot = batch["v_allele"]
                true_start = batch["v_germline_start"]; slen = seg_mask.sum(1)
                in_topk = (topk == prim.unsqueeze(1)).any(dim=1)                     # (B,)
                rr.append(float(in_topk.float().mean()))
                # band recall vs TRUE allele on PREDICTED segment, conditioned on true in top-k
                bl = head(seg_reps, seg_mask, pos_reps[prim], pos_mask[prim], seg_tok, pos_tok[prim])
                centers = _topm_centers(bl, a.w, a.topm)                            # (B,m)
                covered = ((centers - true_start.unsqueeze(1)).abs() <= a.w).any(dim=1)
                if int(in_topk.sum()) > 0:
                    br.append(float(covered[in_topk].float().mean()))
                # fail-open / budget on the true-allele band (speed proxy)
                ev = peak_evidence(bl, seg_tok, pos_tok[prim], seg_mask)
                commit = ev >= a.ev_thresh
                fo.append(float((~commit).float().mean()))
                Lg = bl.shape[-1]
                cols = torch.where(commit, torch.full_like(slen, 2 * a.w + 1), torch.full_like(slen, Lg))
                cb.append(float((cols.float() * slen.float().clamp(min=1)).mean()))
                # DP log-partition rerank over top-k (full-width band), rule-1 reader
                seg_e = seg_reps.unsqueeze(1).expand(B, k, S, seg_reps.shape[-1]).reshape(B * k, S, -1)
                sm_e = seg_mask.unsqueeze(1).expand(B, k, S).reshape(B * k, S)
                st_e = seg_tok.unsqueeze(1).expand(B, k, S).reshape(B * k, S)
                flat = topk.reshape(-1)
                gr, gm, gt = pos_reps[flat], pos_mask[flat], pos_tok[flat]
                ctr = torch.zeros(B * k, dtype=torch.long, device=device)
                Lgmax = gr.shape[1]
                sc = al.alignment_score(seg_e, sm_e, gr, gm, ctr, Lgmax, st_e, gt).reshape(B, k)
                rerank = topk.gather(1, sc.argmax(dim=1, keepdim=True)).squeeze(1)   # (B,) chosen allele
                dp.append(float((multihot.gather(1, rerank.unsqueeze(1)).squeeze(1) > 0).float().mean()))
                t1.append(float((multihot.gather(1, topk[:, :1]).squeeze(1) > 0).float().mean()))
        def m(xs):
            return bootstrap_ci(xs)[0] if xs else float("nan")
        print(f"{cname:18s} {m(rr):10.3f} {m(br):11.3f} {m(t1):10.3f} {m(dp):10.3f} "
              f"{m(fo):9.3f} {m(cb):9.0f}")


if __name__ == "__main__":
    main()
