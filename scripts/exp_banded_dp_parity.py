"""Parity gate for build step 3: on the trained model, the banded SeedExtendAligner (band
centered on the TRUE germline_start, oracle) must produce coordinates that match the full
soft-DP within +-1nt per lattice cell — confirming banding is exact in the new code path
(reproduces the Gate-0 band-sweep result through SeedExtendAligner). No retrain.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_banded_dp_parity.py --n 300 --w 16
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.nn.aligner.banded_dp import SeedExtendAligner
from alignair.nn.aligner.germline_aligner import decode_germline_coords
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--w", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    sd = model.aligner                                   # the trained SoftDPAligner
    al = SeedExtendAligner(d_model=ck["config"]["d_model"]).to(device).eval()
    al.seg_proj.load_state_dict(sd.seg_proj.state_dict())
    al.germ_proj.load_state_dict(sd.germ_proj.state_dict())
    with torch.no_grad():
        for p in ("log_scale", "_gap_open", "_gap_extend", "_del_gap", "_match_weight"):
            getattr(al, p).copy_(getattr(sd, p))
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}

    print(f"banded(oracle w={a.w}) vs full soft-DP coords | start+end within +-1nt")
    print(f"{'cell':18s} {'agree-frac':>12s}")
    for cname in CELLS:
        cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                             "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
        loader = DataLoader(AlignAIRGym([dc], rs, n=a.n, seed=0, curriculum=cur),
                            batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
        agree = tot = 0
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"],
                                                           batch["region_labels"], "V")
                seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
                idx = batch["v_primary_idx"]
                gr = ref_emb["V"]["pos_reps"][idx]; gm = ref_emb["V"]["pos_mask"][idx]
                center = batch["v_germline_start"]
                sl_f, el_f = sd(seg_reps, seg_mask, gr, gm)                       # full soft-DP
                sl_b, el_b = al(seg_reps, seg_mask, gr, gm, center, a.w)          # banded (oracle)
                gs_f, ge_f = decode_germline_coords(sl_f, el_f, soft=True)
                gs_b, ge_b = decode_germline_coords(sl_b, el_b, soft=True)
                ok = ((gs_f - gs_b).abs() <= 1) & ((ge_f - ge_b).abs() <= 1)
                agree += int(ok.sum()); tot += ok.numel()
        print(f"{cname:18s} {agree/max(tot,1):12.3f}")


if __name__ == "__main__":
    main()
