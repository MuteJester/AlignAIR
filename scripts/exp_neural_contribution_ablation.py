"""Neural-contribution ablation (spec §5.1) — INFERENCE-TIME reader comparison on a trained
seed_extend checkpoint. Defends "learned aligner + exact structured decoder, not classical
alignment" by showing the LEARNED DP log-partition reader beats the pooled-retrieval and the
order-light MaxSim readers on allele calling (esp. heavy-SHM-V), on the SAME trained weights.

Per cell, V allele accuracy (predicted allele is in the true multi-hot set) under three readers
over the model's retrieval top-k:
  - pooled   : the model's pooled-cosine retrieval argmax (the cheap baseline)
  - maxsim   : ColBERT token-level MaxSim grid, Σ_i max_j cos(R_i,G_j)  (evidence, order-light)
  - dp       : the exact banded soft-DP log-partition (rule-1 reader)

The training-time ablations (raw DP emissions, raw band, no learned reps, no reliability,
frozen/random encoder) require SEPARATE retrain arms (`train_seed_extend.py` with the toggle) and
are run as a follow-up; this script is the cheap inference-time core.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_neural_contribution_ablation.py --model .private/models/seed_extend_d96.pt
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.stats import bootstrap_ci
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/seed_extend_d96.pt")
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--k", type=int, default=16)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    al = model.aligner   # SeedExtendAligner
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    pos_reps, pos_mask, pos_tok = ref_emb["V"]["pos_reps"], ref_emb["V"]["pos_mask"], ref_emb["V"]["pos_tok"]
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}

    print(f"NEURAL-CONTRIBUTION ABLATION (reader comparison) | top-{a.k}")
    print(f"{'cell':18s} {'pooled':>10s} {'maxsim':>10s} {'dp(learned)':>12s}")
    for cname in CELLS:
        cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                             "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
        loader = DataLoader(AlignAIRGym([dc], rs, n=a.n, seed=0, curriculum=cur),
                            batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        acc = {"pooled": [], "maxsim": [], "dp": []}
        with torch.no_grad():
            for batch in loader:
                batch = {kk: (v.to(device) if torch.is_tensor(v) else v) for kk, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                pred_region = out["region_logits"].argmax(-1)
                seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"], pred_region, "V")
                seg_reps, _ = extract_segment(out["reps"], batch["mask"], pred_region, "V")
                B, S = seg_tok.shape
                topk = out["match"]["V"].topk(min(a.k, pos_reps.shape[0]), dim=-1).indices  # (B,k)
                k = topk.shape[1]
                multihot = batch["v_allele"]
                flat = topk.reshape(-1)
                gr, gm, gt = pos_reps[flat], pos_mask[flat], pos_tok[flat]
                seg_e = seg_reps.unsqueeze(1).expand(B, k, S, seg_reps.shape[-1]).reshape(B * k, S, -1)
                sm_e = seg_mask.unsqueeze(1).expand(B, k, S).reshape(B * k, S)
                st_e = seg_tok.unsqueeze(1).expand(B, k, S).reshape(B * k, S)
                # pooled: the model's own retrieval order (top-1)
                pooled_choice = topk[:, 0]
                # maxsim: token-level Sum_i max_j cos(seg_i, germ_j) over the candidate grid
                Sn = torch.nn.functional.normalize(seg_e, dim=-1)
                Gn = torch.nn.functional.normalize(gr, dim=-1)
                grid = torch.einsum("bid,bjd->bij", Sn, Gn)          # (B*k,S,Lg)
                grid = grid.masked_fill(~gm.unsqueeze(1), -1e4).masked_fill(~sm_e.unsqueeze(-1), 0.0)
                maxsim = grid.max(dim=2).values.sum(dim=1).reshape(B, k)
                maxsim_choice = topk.gather(1, maxsim.argmax(1, keepdim=True)).squeeze(1)
                # dp: exact banded soft-DP log-partition (full-width here; banding ~ full in score)
                ctr = torch.zeros(B * k, dtype=torch.long, device=device)
                sc = al.alignment_score(seg_e, sm_e, gr, gm, ctr, gr.shape[1], st_e, gt).reshape(B, k)
                dp_choice = topk.gather(1, sc.argmax(1, keepdim=True)).squeeze(1)
                for name, choice in (("pooled", pooled_choice), ("maxsim", maxsim_choice), ("dp", dp_choice)):
                    acc[name].append(float((multihot.gather(1, choice.unsqueeze(1)).squeeze(1) > 0).float().mean()))
        def m(xs):
            return bootstrap_ci(xs)[0] if xs else float("nan")
        print(f"{cname:18s} {m(acc['pooled']):10.3f} {m(acc['maxsim']):10.3f} {m(acc['dp']):12.3f}")
    print("\nDEFENSE: if dp(learned) beats pooled/maxsim by a CI-disjoint margin on heavy_shm_fulllen,"
          "\n  the structured neural reader does real work (NOT a repackaged classical aligner).")


if __name__ == "__main__":
    main()
