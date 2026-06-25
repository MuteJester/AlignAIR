"""Diagnose WHY ~3% of junction_boundary V-start bands miss. Head-free: computes the RAW
base-match diagonal argmax (the band head's dominant feature) per read and characterizes the
reads whose raw band center is >tol off the true germline_start, vs the reads that hit.

Answers: is the SIGNAL ambiguous on those reads (data: short V / weak true-diagonal peak /
trimmed start) or would a head fix it? Run:
  PYTHONPATH=src ./.venv/bin/python scripts/diag_junction_misses.py --n 600 --tol 16
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.nn.band_head import base_match_matrix
from alignair.nn.pointer_aligner import weighted_leading_diag
from torch.utils.data import DataLoader


def _cell_loader(dc, rs, cell_params, n, bs, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(cell_params),
                         "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
    return DataLoader(AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur),
                      batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--tol", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    rs = ReferenceSet.from_dataconfigs(dc)
    ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(seed=0); cells = {c.name: c for c in lat.cells}

    for cname in ("clean", "junction_boundary"):
        rows = {"seg_len": [], "true": [], "err": [], "mut": [], "true_weaker": []}
        n_hit = n_miss = 0
        loader = _cell_loader(dc, rs, lat.cell_params(cells[cname]), a.n, a.batch_size, 0)
        with torch.no_grad():
            for batch in loader:
                batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"],
                                                           batch["region_labels"], "V")
                idx = batch["v_primary_idx"]
                germ_tok = ref_emb["V"]["pos_tok"][idx]; germ_mask = ref_emb["V"]["pos_mask"][idx]
                bm = base_match_matrix(seg_tok, germ_tok).float()
                w = seg_mask.float().unsqueeze(-1)
                diag = weighted_leading_diag(bm, w).masked_fill(~germ_mask, -1e4)   # (B,Lg)
                pred = diag.argmax(-1)
                true = batch["v_germline_start"]
                err = (pred - true).abs()
                slen = seg_mask.sum(1)
                mut = batch["mutation_rate"].reshape(-1) if "mutation_rate" in batch else torch.zeros_like(err).float()
                B = pred.shape[0]
                ar = torch.arange(B, device=device)
                score_true = diag[ar, true.clamp(0, diag.shape[1] - 1)]
                score_pred = diag[ar, pred]
                miss = err > a.tol
                n_hit += int((~miss).sum()); n_miss += int(miss.sum())
                if miss.any():
                    rows["seg_len"] += slen[miss].tolist()
                    rows["true"] += true[miss].tolist()
                    rows["err"] += err[miss].tolist()
                    rows["mut"] += mut[miss].tolist()
                    rows["true_weaker"] += (score_true[miss] < score_pred[miss] - 1e-3).tolist()
                if not miss.all():
                    pass
        tot = n_hit + n_miss
        print(f"\n=== {cname}: miss rate {n_miss/max(tot,1):.3f} ({n_miss}/{tot}) at tol {a.tol} ===")
        if n_miss:
            import statistics as st
            def ms(x): return f"mean {st.mean(x):.1f} median {st.median(x):.1f} min {min(x):.0f} max {max(x):.0f}"
            print(f"  MISSED reads: seg_len  {ms(rows['seg_len'])}")
            print(f"  MISSED reads: err(nt)  {ms(rows['err'])}")
            print(f"  MISSED reads: true_start {ms(rows['true'])}")
            print(f"  MISSED reads: mut_rate mean {st.mean(rows['mut']):.3f}")
            tw = sum(rows["true_weaker"]) / len(rows["true_weaker"])
            print(f"  MISSED reads: fraction where TRUE diagonal is a WEAKER base-match peak than pred: {tw:.2f}")
            print(f"    (high => SIGNAL is ambiguous/wrong at truth; low => a head COULD recover it)")


if __name__ == "__main__":
    main()
