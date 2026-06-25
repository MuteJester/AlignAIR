"""Gate 1 (geometry gate, spec §5): freeze the 8h soft-DP model, train ONLY the structural
band head on TRUE-region / TRUE-allele V segments, then report per-lattice-cell band recall
(top-1 + top-m union), fail-open rate, and DP cell budget with bootstrap CIs at w=8,16.

HARD STOP: if top-m union recall misses >0.5-1% at w=16, OR the cell budget erases the speed
win, do NOT proceed to kernel/encoder work (spec build order step 1).

Run:
  PYTHONPATH=src ./.venv/bin/python scripts/exp_band_recall_gate.py --train-steps 3000 --n 500
"""
import argparse

import torch
import GenAIRR.data as gdata

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR, extract_segment_tokens
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.curriculum import StratifiedCurriculum
from alignair.gym.instrument.lattice import FrozenLattice
from alignair.gym.instrument.stats import bootstrap_ci
from alignair.nn.band_head import BandHead, band_offset_loss, peak_evidence
from alignair.gym.instrument import band_metrics as BM
from torch.utils.data import DataLoader

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def _segment_inputs(model, batch, ref_emb, device):
    """True-region/true-allele V segment reps + tokens + germline reps + true start."""
    out = model(batch["tokens"], batch["mask"], ref_emb, orientation_ids=batch["orientation_id"])
    canon = out["canon_tokens"]
    seg_tok, seg_mask = extract_segment_tokens(canon, batch["mask"], batch["region_labels"], "V")
    seg_reps = model.germline_encoder.forward_positions(seg_tok, seg_mask)
    idx = batch["v_primary_idx"]
    germ_reps = ref_emb["V"]["pos_reps"][idx]; germ_mask = ref_emb["V"]["pos_mask"][idx]
    germ_tok = ref_emb["V"]["pos_tok"][idx]
    return (seg_reps, seg_mask, germ_reps, germ_mask, seg_tok, germ_tok,
            batch["v_germline_start"], seg_mask.sum(1))


def _cell_loader(dc, rs, cell_params, n, batch_size, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(cell_params),
                         "describe": lambda s, p=0.0: "cell", "stage": lambda s, p=0.0: 0})()
    gym = AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur)
    return DataLoader(gym, batch_size=batch_size,
                      collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def _mixture_loader(dc, rs, n, batch_size, seed):
    """Training stream: the StratifiedCurriculum mixture (covers all difficulty regimes) so
    the band head GENERALIZES across cells rather than overfitting one regime."""
    gym = AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=StratifiedCurriculum())
    return DataLoader(gym, batch_size=batch_size,
                      collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--train-steps", type=int, default=2000)
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--widths", default="8,16")
    ap.add_argument("--topm", type=int, default=2)
    ap.add_argument("--fail-open-thresh", type=float, default=0.1)
    ap.add_argument("--ev-thresh", type=float, default=0.5)   # overlap-fraction fail-open threshold
    a = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    widths = [int(x) for x in a.widths.split(",")]
    dc = gdata.HUMAN_IGH_OGRDB

    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    model = DNAlignAIR(DNAlignAIRConfig(**ck["config"]))
    model.load_state_dict(ck["model"]); model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)                                  # FREEZE the 8h model
    rs = ReferenceSet.from_dataconfigs(dc)
    ref_emb = model.encode_reference(rs)
    lat = FrozenLattice.standard(seed=0)
    cells = {c.name: c for c in lat.cells}

    head = BandHead(d_model=ck["config"]["d_model"]).to(device).train()
    opt = torch.optim.Adam(head.parameters(), lr=1e-3)

    # train the head on the StratifiedCurriculum MIXTURE (generalizes across regimes)
    train_loader = _mixture_loader(dc, rs, a.n * 50, a.batch_size, seed=1)
    it = iter(train_loader); step = 0
    while step < a.train_steps:
        try:
            batch = next(it)
        except StopIteration:
            it = iter(train_loader); batch = next(it)
        batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
        with torch.no_grad():
            inp = _segment_inputs(model, batch, ref_emb, device)
        logits = head(*inp[:6])
        loss = band_offset_loss(logits, inp[6])        # fail-open is a physical evidence threshold, not learned
        opt.zero_grad(); loss.backward(); opt.step()
        step += 1
        if step % 250 == 0:
            print(f"[train] step {step}/{a.train_steps} loss {float(loss):.3f}", flush=True)

    # evaluate per frozen-lattice cell
    head.eval()
    print(f"\nGate 1 band recall (frozen model, true region/allele) | top-m={a.topm} "
          f"ev-thresh={a.ev_thresh}")
    for w in widths:
        print(f"\n--- w={w} (commit = overlap-frac >= {a.ev_thresh}) ---")
        print(f"{'cell':18s} {'union-recall':>14s} {'committed-rec':>16s} {'fail-open':>10s} {'budget':>10s}")
        for cname in CELLS:
            loader = _cell_loader(dc, rs, lat.cell_params(cells[cname]), a.n, a.batch_size, seed=0)
            tm, cr, fo, cb = [], [], [], []
            with torch.no_grad():
                for batch in loader:
                    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
                    inp = _segment_inputs(model, batch, ref_emb, device)
                    logits = head(*inp[:6]); true = inp[6]; slen = inp[7]
                    # fail-open on the physical overlap-fraction evidence (spurious low-overlap
                    # peaks of signal-absent reads route to full DP). pseudo-conf reuses the
                    # committed-recall metric: sigmoid((ev - thresh)*10) >= 0.5  <=>  ev >= thresh.
                    ev = peak_evidence(logits, inp[4], inp[5], inp[1])
                    conf = (ev - a.ev_thresh) * 10.0
                    tm.append(BM.topm_union_recall(logits, true, w, a.topm))
                    cr.append(BM.committed_recall(logits, conf, true, w, a.topm, 0.5))
                    fo.append(BM.conf_fail_open_rate(conf, 0.5))
                    Lg = logits.shape[-1]; commit = torch.sigmoid(conf) >= 0.5
                    cols = torch.where(commit, torch.full_like(slen, 2 * w + 1), torch.full_like(slen, Lg))
                    cb.append(float((cols.float() * slen.float()).mean()))
            def lo(xs):
                m, l, h = bootstrap_ci(xs); return m, l
            (mm, lm), (mc, lc), (mf, lf), (mb, lb) = lo(tm), lo(cr), lo(fo), lo(cb)
            print(f"{cname:18s} {mm:.3f}[lo {lm:.3f}] {mc:.3f}[lo {lc:.3f}] "
                  f"{mf:8.3f} {mb:10.0f}")


if __name__ == "__main__":
    main()
