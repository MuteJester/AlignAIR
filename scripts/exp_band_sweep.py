"""De-risking experiment for the parallel-scan pivot: does restricting the EXACT soft-DP
to a BAND of width w (around the true alignment diagonal) preserve coordinate accuracy?

If banded-DP @ w=32-64 ~= full-DP, then a learned-band + fused-banded-scan keeps the
soft-DP's accuracy at O(w*L) cost -> the whole "exact DP but fast" thesis holds. Uses the
8h-trained soft-DP (scaled_long.pt) so M is realistic; band is centered on GROUND-TRUTH
germline_start (oracle band) to isolate the banding question from band-prediction.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_band_sweep.py --n 300 --tol 1.0
"""
import argparse

import torch
import torch.nn.functional as F
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.nn.soft_dp_aligner import soft_dp_end_logits, _reverse_valid_2d, NEG
from torch.utils.data import DataLoader

CELLS = ("clean", "junction_boundary", "heavy_shm_fulllen", "indel")


def banded_coords(aligner, seg_reps, seg_mask, germ_reps, germ_mask, true_start, w):
    """Replicate SoftDPAligner.forward but mask M to a band |j-(start+i)|<=w (None=full)."""
    go = -F.softplus(aligner._gap_open); ge = -F.softplus(aligner._gap_extend)
    dg = -F.softplus(aligner._del_gap)
    M = aligner._scores(seg_reps, germ_reps)                      # (B,S,Lg)
    if w is not None:
        B, S, Lg = M.shape
        i = torch.arange(S, device=M.device)[None, :, None]
        j = torch.arange(Lg, device=M.device)[None, None, :]
        center = true_start.view(-1, 1, 1) + i                    # (B,S,1)
        band = (j - center).abs() <= w
        M = M.masked_fill(~band, NEG)
    end = soft_dp_end_logits(M, seg_mask, germ_mask, go, ge, dg)
    seg_len = seg_mask.sum(1); germ_len = germ_mask.sum(1)
    Mr = _reverse_valid_2d(M.transpose(1, 2), germ_len).transpose(1, 2)
    Mr = _reverse_valid_2d(Mr, seg_len)
    end_rev = soft_dp_end_logits(Mr, seg_mask, germ_mask, go, ge, dg)
    start = _reverse_valid_2d(end_rev, germ_len).masked_fill(~germ_mask, NEG)
    end = end.masked_fill(~germ_mask, NEG)
    return start.argmax(-1), end.argmax(-1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--tol", type=float, default=1.0)
    ap.add_argument("--widths", default="8,16,32,64")
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    widths = [int(x) for x in a.widths.split(",")] + [None]       # None = full DP

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    lat = FrozenLattice.standard(seed=0)
    cells = {c.name: c for c in lat.cells}

    print(f"coord-acc within +-{a.tol}nt | full-DP vs banded-DP (oracle band) | V start+end")
    hdr = f"{'cell':18s}" + "".join(f"{('w='+str(w) if w else 'FULL'):>8s}" for w in widths)
    print(hdr)
    for cname in CELLS:
        cell = cells[cname]
        cur = type("C", (), {"params": lambda s, p=0.0: dict(lat.cell_params(cell)),
                             "describe": lambda s, p=0.0: cname, "stage": lambda s, p=0.0: 0})()
        gym = AlignAIRGym([gdata.HUMAN_IGH_OGRDB], rs, n=a.n, seed=0, curriculum=cur)
        loader = DataLoader(gym, batch_size=a.batch_size,
                            collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        ref_emb = model.encode_reference(rs)
        hits = {w: 0 for w in widths}; tot = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb,
                            orientation_ids=batch["orientation_id"])
                canon = out["canon_tokens"]
                rl = batch["region_labels"]                       # teacher-forced region
                seg_tok, seg_mask = extract_segment_tokens(canon, batch["mask"], rl, "V")
                seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
                idx = batch["v_primary_idx"]
                germ_reps = ref_emb["V"]["pos_reps"][idx]; germ_mask = ref_emb["V"]["pos_mask"][idx]
                gs_t = batch["v_germline_start"]; ge_t = batch["v_germline_end"]
                for w in widths:
                    gs, ge = banded_coords(model.aligner, seg_reps, seg_mask, germ_reps,
                                           germ_mask, gs_t, w)
                    ok = ((gs - gs_t).abs() <= a.tol) & ((ge - ge_t).abs() <= a.tol)
                    hits[w] += int(ok.sum())
                tot += seg_mask.shape[0]
        row = f"{cname:18s}" + "".join(f"{hits[w]/max(tot,1):8.3f}" for w in widths)
        print(row)


if __name__ == "__main__":
    main()
