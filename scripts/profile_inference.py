"""Profile DNAlignAIR inference on ~5K reads to locate the time bottleneck.

Two complementary views, because cProfile alone LIES on GPU code (CUDA kernels are
async — their cost surfaces at the next .cpu()/sync, mis-attributed to whatever
Python frame happens to sync). So we report:

  (1) STAGE breakdown: the predict_reads pipeline re-implemented inline with
      torch.cuda.synchronize() around each phase → true wall-clock per phase
      (tokenize / forward / region-decode / germline-coords / soft-DP rerank /
      python post-loop). This is the number that answers "is soft-DP the bottleneck".

  (2) cProfile over the real predict_reads() call → Python-level hot functions,
      useful for the CPU-bound parts (the per-read python loop, Biopython, etc.).

Run:
  .venv/bin/python scripts/profile_inference.py --n 5000
  .venv/bin/python scripts/profile_inference.py --n 5000 --rerank none   # soft-DP off, for contrast
"""
import argparse
import cProfile
import io
import os
import pstats
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import gen_records  # noqa: E402

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR, extract_segment, extract_segment_tokens  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402
from alignair.data.tokenizer import pad_tokenize  # noqa: E402
from alignair.nn.heads.region import decode_boundaries  # noqa: E402
from alignair.nn.aligner.germline_aligner import decode_germline_coords  # noqa: E402
from alignair.training.germline_tf import compute_germline_logits  # noqa: E402
from alignair.nn.heads.state import state_reliability  # noqa: E402


class Stage:
    """Wall-clock accumulator with CUDA sync so async kernels are charged correctly."""
    def __init__(self, cuda):
        self.t = {}
        self.cuda = cuda

    def sync(self):
        if self.cuda:
            torch.cuda.synchronize()

    def add(self, name, dt):
        self.t[name] = self.t.get(name, 0.0) + dt

    def report(self, total):
        print(f"\n{'STAGE':28s} {'sec':>9} {'%':>7} {'ms/read':>9}")
        print("-" * 56)
        for k, v in sorted(self.t.items(), key=lambda x: -x[1]):
            print(f"{k:28s} {v:9.3f} {v/total*100:6.1f}% {v/N*1000:9.3f}")
        print("-" * 56)
        print(f"{'TOTAL (sum of stages)':28s} {total:9.3f}")


def staged_predict(model, rs, reads, device, topk, rerank, batch_size, st):
    """Inlined predict_reads with per-phase timers. Mirrors the real path's heavy ops."""
    model.eval()
    t = time.perf_counter(); st.sync()
    ref_emb = model.encode_reference(rs)
    st.sync(); st.add("encode_reference (1x)", time.perf_counter() - t)

    has_d = rs.has_d
    genes = ["v", "j"] + (["d"] if has_d else [])
    n_allowed = {g.upper(): len(rs.gene(g.upper()).names) for g in genes}

    for s in range(0, len(reads), batch_size):
        chunk = reads[s:s + batch_size]

        t = time.perf_counter()
        tokens, mask = pad_tokenize(chunk)
        tokens, mask = tokens.to(device), mask.to(device)
        st.sync(); st.add("tokenize+H2D", time.perf_counter() - t)

        t = time.perf_counter(); st.sync()
        out = model(tokens, mask, ref_emb, candidate_masks=None)
        st.sync(); st.add("model.forward (backbone+heads)", time.perf_counter() - t)

        canon = out["canon_tokens"]

        t = time.perf_counter(); st.sync()
        dec = decode_boundaries(out["region_logits"], mask, has_d=has_d)
        pred_region = out["region_logits"].argmax(-1)
        pred_idx = {g.upper(): out["match"][g.upper()].argmax(-1) for g in genes}
        topk_idx = {g.upper(): out["match"][g.upper()].topk(
            min(topk, n_allowed[g.upper()]), dim=-1).indices for g in genes}
        st.sync(); st.add("region/topk decode", time.perf_counter() - t)

        t = time.perf_counter(); st.sync()
        gl = compute_germline_logits(model, canon, mask, {}, ref_emb, has_d,
                                     region_labels=pred_region, allele_idx=pred_idx)
        gcoord = {g: decode_germline_coords(gl[g][0], gl[g][1]) for g in genes}
        st.sync(); st.add("germline coords", time.perf_counter() - t)

        if rerank == "learned":
            for g in genes:
                G = g.upper()
                t = time.perf_counter(); st.sync()
                seg_tok, seg_mask = extract_segment_tokens(canon, mask, pred_region, G)
                seg_pos = model.germline_encoder.forward_positions(seg_tok, seg_mask)
                seg_state, _ = extract_segment(out["state_logits"], mask, pred_region, G)
                seg_rel = state_reliability(seg_state)
                st.sync(); st.add("rerank: segment encode", time.perf_counter() - t)

                t = time.perf_counter(); st.sync()
                pos_reps, pos_mask = ref_emb[G]["pos_reps"], ref_emb[G]["pos_mask"]
                pos_tok = ref_emb[G]["pos_tok"]
                B = len(chunk)
                ti = topk_idx[G]
                kk = ti.shape[1]
                read_ix = torch.arange(B, device=device).repeat_interleave(kk)
                cand_ix = ti.reshape(-1)
                parts = []
                for a in range(0, B * kk, 2048):
                    sl = slice(a, a + 2048)
                    ri, ci = read_ix[sl], cand_ix[sl]
                    parts.append(model.aligner.alignment_score(
                        seg_pos[ri], seg_mask[ri], pos_reps[ci], pos_mask[ci],
                        seg_tok=seg_tok[ri], germ_tok=pos_tok[ci],
                        seg_reliability=seg_rel[ri]))
                sc_all = torch.cat(parts).reshape(B, kk)
                _ = sc_all.argmax(dim=1).cpu().tolist()  # force sync like real path
                st.sync(); st.add(f"rerank: soft-DP align [{G}]", time.perf_counter() - t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--rerank", default="learned", choices=["learned", "none"])
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()

    global N
    N = args.n
    device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    cuda = device == "cuda"
    print(f"device={device}  n={args.n}  topk={args.topk}  batch={args.batch_size}  rerank={args.rerank}")

    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"])
    m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()

    print(f"generating {args.n} reads ...")
    recs = gen_records(1.0, args.n, args.seed, None, overrides={"mutation_rate": 0.20})
    reads = [r["sequence"] for r in recs]

    # warmup (CUDA autotune, lazy init, allocator)
    print("warmup ...")
    with torch.no_grad():
        predict_reads(m, rs, reads[:128], device=device, topk=args.topk, rerank=args.rerank)

    # ---- (1) stage breakdown (synchronized) ----
    print("\n=== (1) SYNCHRONIZED STAGE BREAKDOWN ===")
    st = Stage(cuda)
    t0 = time.perf_counter(); st.sync()
    with torch.no_grad():
        staged_predict(m, rs, reads, device, args.topk, args.rerank, args.batch_size, st)
    st.sync(); wall = time.perf_counter() - t0
    st.report(sum(st.t.values()))
    print(f"\nWALL CLOCK (staged path): {wall:.3f}s   throughput: {args.n/wall:.1f} reads/s")

    # ---- (2) cProfile over the REAL predict_reads ----
    print("\n=== (2) cPROFILE over real predict_reads (cumulative) ===")
    pr = cProfile.Profile()
    t0 = time.perf_counter()
    with torch.no_grad():
        pr.runcall(predict_reads, m, rs, reads, device=device,
                   topk=args.topk, rerank=args.rerank, batch_size=args.batch_size)
    if cuda:
        torch.cuda.synchronize()
    wall = time.perf_counter() - t0
    print(f"WALL CLOCK (real predict_reads): {wall:.3f}s   throughput: {args.n/wall:.1f} reads/s\n")
    s = io.StringIO()
    ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
    ps.print_stats(30)
    print(s.getvalue())
    print("\n=== cPROFILE by TOTAL time (self, not cumulative) ===")
    s2 = io.StringIO()
    pstats.Stats(pr, stream=s2).sort_stats("tottime").print_stats(20)
    print(s2.getvalue())


if __name__ == "__main__":
    main()
