"""Layer-1 de-risk: does classical raw-base rescore of the trained XAttnAligner's neural top-k V
candidates crack heavy-SHM / sibling V discrimination, with NO retrain? Per FrozenLattice cell,
compares V top-1 set-accuracy of the neural matcher vs the classical rescore of its own top-k,
and reports the pool recall (the rescore ceiling).

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python \
      scripts/exp_xattn_rescore.py --model .private/models/xattn_igh.pt --n 400 --topk 16
"""
import argparse
import torch
import GenAIRR.data as gdata
from torch.utils.data import DataLoader

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.nn.heads.region import decode_boundaries
from alignair.inference.wfa_caller import call_segment
from alignair.align import SeedPrefilter, get_aligner
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice

CELLS = ("clean", "heavy_shm_fulllen", "indel", "junction_boundary")


def _loader(dc, rs, cell, lat, n, bs, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(lat.cell_params(cell)),
                         "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
    return DataLoader(AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur),
                      batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def _detok(tok_row, mask_row):
    # canonical read string from the model's canon_tokens (already orientation-canonicalized)
    from alignair.data.tokenizer import TOKEN_DICT
    inv = {v: k for k, v in TOKEN_DICT.items()}
    return "".join(inv.get(int(t), "N") for t, ok in zip(tok_row.tolist(), mask_row.tolist()) if ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/xattn_igh.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    ap.add_argument("--topk", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    m = XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    with torch.no_grad():
        ref = m.encode_reference(rs)
    sp, al = SeedPrefilter(rs, k=11), get_aligner()
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}
    print(f"model {a.model} (step {ck.get('step')}) | topk={a.topk} | aligner={type(al).__name__}\n")
    print(f"{'cell':20s} {'neural_V':>9s} {'rescore_V':>10s} {'pool_recall':>12s}")
    for cn in CELLS:
        nh = rh = pr = tot = 0
        with torch.no_grad():
            for b in _loader(dc, rs, cells[cn], lat, a.n, a.bs, 7):
                b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                o = m(b["tokens"], b["mask"], ref, topk=a.topk, seed_m=0, cand_chunk=2)
                mh = b["v_allele"]
                neural_best = o["match"]["V"]["best_global_idx"]
                pool = o["match"]["V"]["pool_idx"]                       # (B,k) neural top-k
                dec = decode_boundaries(o["region_logits"], b["mask"], has_d=rs.has_d)
                canon = [_detok(o["canon_tokens"][i], b["mask"][i]) for i in range(neural_best.shape[0])]
                for i in range(neural_best.shape[0]):
                    nh += int(mh[i, neural_best[i]] > 0)
                    pr += int((mh[i].index_select(0, pool[i]) > 0).any())
                    vs, ve = dec[i]["v_start"], dec[i]["v_end"]          # region-derived V segment
                    seg = canon[i][vs:ve] if (vs is not None and ve and ve > vs) else ""
                    sc = call_segment(seg, "V", pool[i].cpu().tolist(), rs, sp, al, m_seed=0)
                    rh += int(sc is not None and mh[i, sc.best_idx] > 0)
                    tot += 1
        print(f"{cn:20s} {nh/tot:9.3f} {rh/tot:10.3f} {pr/tot:12.3f}")
    print("\nrescore_V > neural_V on heavy_shm => classical rescue works (Layer 1 wins).")
    print("rescore_V ~ pool_recall => rescore is near-optimal given the pool (ceiling = retrieval).")


if __name__ == "__main__":
    main()
