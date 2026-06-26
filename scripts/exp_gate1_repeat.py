"""Gate-1 REPEAT on the CO-TRAINED shared encoder (spec build step 4): unlike the original Gate 1
(which trained a fresh band head on the FROZEN 8h soft-DP), this loads a trained seed_extend
checkpoint and evaluates the model's OWN co-trained band head + peak_evidence, on true-region/
true-allele V segments. Tells us whether shared-encoder co-training LIFTS junction past its 0.97
frozen-proxy lower bound while preserving clean/heavy/indel.

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_gate1_repeat.py --model .private/models/seed_extend_d96.pt
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
from alignair.gym.instrument.band_metrics import topm_union_recall, committed_recall, conf_fail_open_rate
from alignair.nn.aligner.band_head import peak_evidence
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/seed_extend_d96.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--topm", type=int, default=2)
    ap.add_argument("--ev-thresh", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dc = gdata.HUMAN_IGH_OGRDB
    widths = [int(x) for x in a.widths.split(",")]
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    assert getattr(model, "seed_extend", False) and hasattr(model, "band_head"), "need a seed_extend ckpt"
    rs = ReferenceSet.from_dataconfigs(dc); ref_emb = model.encode_reference(rs)
    pos_reps, pos_mask, pos_tok = ref_emb["V"]["pos_reps"], ref_emb["V"]["pos_mask"], ref_emb["V"]["pos_tok"]
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}

    print(f"GATE-1 REPEAT (co-trained band head) | top-m={a.topm} ev>={a.ev_thresh}")
    for w in widths:
        print(f"\n--- w={w} ---")
        print(f"{'cell':18s} {'union-rec':>12s} {'committed':>12s} {'fail-open':>10s} {'budget':>9s}")
        for cname in CELLS:
            cur = type("C", (), {"params": lambda s, p=0.0, c=cells[cname]: dict(lat.cell_params(c)),
                                 "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
            loader = DataLoader(AlignAIRGym([dc], rs, n=a.n, seed=0, curriculum=cur),
                                batch_size=a.batch_size, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))
            tm, cr, fo, cb = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
                    seg_tok, seg_mask = extract_segment_tokens(out["canon_tokens"], batch["mask"],
                                                               batch["region_labels"], "V")
                    seg_reps, _ = extract_segment(out["reps"], batch["mask"], batch["region_labels"], "V")
                    idx = batch["v_primary_idx"]; true = batch["v_germline_start"]; slen = seg_mask.sum(1)
                    bl = model.band_head(seg_reps, seg_mask, pos_reps[idx], pos_mask[idx], seg_tok, pos_tok[idx])
                    ev = peak_evidence(bl, seg_tok, pos_tok[idx], seg_mask)
                    conf = (ev - a.ev_thresh) * 10.0
                    tm.append(topm_union_recall(bl, true, w, a.topm))
                    cr.append(committed_recall(bl, conf, true, w, a.topm, 0.5))
                    fo.append(conf_fail_open_rate(conf, 0.5))
                    Lg = bl.shape[-1]; commit = torch.sigmoid(conf) >= 0.5
                    cols = torch.where(commit, torch.full_like(slen, 2 * w + 1), torch.full_like(slen, Lg))
                    cb.append(float((cols.float() * slen.float().clamp(min=1)).mean()))
            def lo(xs):
                m, l, h = bootstrap_ci(xs); return m, l
            (mm, lm), (mc, lc), (mf, _), (mb, _) = lo(tm), lo(cr), lo(fo), lo(cb)
            print(f"{cname:18s} {mm:.3f}[lo {lm:.3f}] {mc:.3f}[lo {lc:.3f}] {mf:8.3f} {mb:9.0f}")
    print("\nGREEN if committed-recall >= the frozen-proxy Gate 1 (clean/heavy/indel ~1.0, junction ~0.97),"
          "\n  ideally junction LIFTS past 0.97 thanks to co-training.")


if __name__ == "__main__":
    main()
