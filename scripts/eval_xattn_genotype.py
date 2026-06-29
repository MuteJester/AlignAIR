"""Dynamic-genotype evaluation of a trained XAttnAligner: (1) does restricting inference to a
donor's PARTIAL genotype (a subset of the trained reference) improve allele accuracy — especially
on heavy-SHM, where confusable siblings drive errors; (2) can the model call NOVEL alleles (germline
strings never in training) when they are added to the reference at inference.

Run:
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=src .venv/bin/python \
      scripts/eval_xattn_genotype.py --model .private/models/xattn_igh.pt --n 400
"""
import argparse
import random

import torch
import GenAIRR.data as gdata
from torch.utils.data import DataLoader

from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.reference.reference_set import ReferenceSet
from alignair.gym import AlignAIRGym, gym_collate
from alignair.gym.instrument.lattice import FrozenLattice


def _loader(dc, rs, cell, lat, n, bs, seed):
    cur = type("C", (), {"params": lambda s, p=0.0: dict(lat.cell_params(cell)),
                         "describe": lambda s, p=0.0: "c", "stage": lambda s, p=0.0: 0})()
    return DataLoader(AlignAIRGym([dc], rs, n=n, seed=seed, curriculum=cur),
                      batch_size=bs, collate_fn=lambda b: gym_collate(b, rs, rs.has_d))


def _perturb(seq, k, rng):
    s = list(seq.upper())
    pos = [i for i, c in enumerate(s) if c in "ACGT"]
    rng.shuffle(pos)
    for i in pos[:k]:
        s[i] = rng.choice([b for b in "ACGT" if b != s[i]])
    return "".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/xattn_igh.pt")
    ap.add_argument("--n", type=int, default=400)
    ap.add_argument("--bs", type=int, default=16)
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.model, map_location="cpu", weights_only=False)
    m = XAttnAligner(DNAlignAIRConfig(**ck["config"])); m.load_state_dict(ck["model"]); m.to(dev).eval()
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    with torch.no_grad():
        ref = m.encode_reference(rs)
    lat = FrozenLattice.standard(0); cells = {c.name: c for c in lat.cells}
    vnames = rs.gene("V").names
    rng = random.Random(0)

    # ---------- Part 1: partial genotype ----------
    print("=== PART 1: partial genotype (align reads against a donor's allele SUBSET) ===")
    print("for reads whose true V is in the donor genotype, V acc vs the FULL reference\n")
    print(f"{'cell':18s} {'genotype':>10s} {'V_acc':>7} {'n_reads':>8}")
    for cn in ("clean", "heavy_shm_fulllen"):
        cell = cells[cn]
        for frac in (1.00, 0.50, 0.25):
            # a fixed donor genotype = random `frac` of V alleles (one donor for this condition)
            kV = max(1, int(len(vnames) * frac))
            donor = set(random.Random(7).sample(range(len(vnames)), kV))
            cmask = torch.zeros(len(vnames), dtype=torch.bool)
            for i in donor:
                cmask[i] = True
            cmask = cmask.to(dev)
            hit = tot = 0
            with torch.no_grad():
                for b in _loader(dc, rs, cell, lat, a.n, a.bs, 5):
                    b = {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in b.items()}
                    keep = torch.tensor([int(i) in donor for i in b["v_primary_idx"].tolist()],
                                        device=dev)
                    if keep.sum() == 0:
                        continue
                    cm = {"V": cmask} if frac < 1.0 else None
                    o = m(b["tokens"], b["mask"], ref, candidate_masks=cm, seed_m=0, cand_chunk=2)
                    best = o["match"]["V"]["best_global_idx"]; mh = b["v_allele"]
                    ok = (mh.gather(1, best[:, None]).squeeze(1) > 0) & keep
                    hit += int(ok.sum()); tot += int(keep.sum())
            label = "full" if frac == 1.0 else f"{int(frac*100)}%"
            print(f"{cn:18s} {label:>10s} {hit/max(tot,1):7.3f} {tot:8d}")

    # ---------- Part 2: novel alleles ----------
    print("\n=== PART 2: novel alleles (germlines NEVER in training, added to the reference) ===")
    print("read = a novel V germline (real allele perturbed by k SNPs) + a real J; align vs the")
    print("AUGMENTED reference (full + novel). Recall = top-1 call is the novel allele.\n")
    Vg, Jg = rs.gene("V"), rs.gene("J")
    jseq = Jg.sequences[0]
    for ksnp in (3, 6, 10):
        # build augmented reference with 30 novel V alleles (perturbed parents)
        parents = random.Random(1).sample(range(len(vnames)), 30)
        novelV = {}
        full_v = {Vg.names[i]: Vg.sequences[i] for i in range(len(vnames))}
        novel_names = []
        for pi in parents:
            nn = f"{Vg.names[pi]}__NOVEL{ksnp}"
            full_v[nn] = _perturb(Vg.sequences[pi], ksnp, rng)
            novelV[nn] = full_v[nn]; novel_names.append(nn)
        genes = {"V": full_v, "D": {rs.gene("D").names[i]: rs.gene("D").sequences[i] for i in range(len(rs.gene("D").names))},
                 "J": {Jg.names[i]: Jg.sequences[i] for i in range(len(Jg.names))}}
        ars = ReferenceSet.from_genotype(genes)
        with torch.no_grad():
            aref = m.encode_reference(ars)
        anames = ars.gene("V").names
        reads = [novelV[nn] + jseq for nn in novel_names]            # novel V..J reads
        hit1 = hitset = 0
        from alignair.data.tokenizer import pad_tokenize
        with torch.no_grad():
            for s in range(0, len(reads), a.bs):
                chunk = reads[s:s + a.bs]
                tok, msk = pad_tokenize(chunk); tok, msk = tok.to(dev), msk.to(dev)
                o = m(tok, msk, aref, seed_m=8, reference_set=ars, cand_chunk=2)
                best = o["match"]["V"]["best_global_idx"].tolist()
                pool = o["match"]["V"]["pool_idx"].tolist()
                for j, nn in enumerate(novel_names[s:s + a.bs]):
                    tgt = anames.index(nn)
                    hit1 += int(anames[best[j]] == nn)
                    hitset += int(tgt in pool[j])
        print(f"  novel @ {ksnp:2d} SNPs:  top1_recall={hit1/len(reads):.3f}  in_pool(admitted)={hitset/len(reads):.3f}  (n=30)")


if __name__ == "__main__":
    main()
