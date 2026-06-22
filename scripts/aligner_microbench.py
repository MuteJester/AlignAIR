"""Phase-0 microbenchmark: can a classical aligner do the V rerank fast enough?
Times parasail / edlib / Biopython on ~realistic V-segment-vs-germline local alignments,
projects to the full fixture (4400 reads x topk candidates), and compares to the GPU soft-DP
rerank cost (~10s for V at 5000 reads, from scripts/profile_inference.py)."""
import sys, os, time, random
sys.path.insert(0, os.path.dirname(__file__))
import json
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet

N_READS = 4400
TOPK = 16
PAIRS = N_READS * TOPK


def main():
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    vref = rs.gene("V")
    germs = [s for s in vref.sequences]
    # realistic V segments: take canonical seqs from the fixture, slice truth V coords
    segs = []
    for l in open("experiments/headtohead/cases.jsonl"):
        c = json.loads(l); r = c["record"]
        vs, ve = r.get("v_sequence_start"), r.get("v_sequence_end")
        if c.get("orientation_id", 0) == 0 and vs is not None and ve and ve - vs > 50:
            segs.append(c["sequence"][vs:ve].upper())
        if len(segs) >= 500:
            break
    # fixed candidate set per read = 16 germlines (timing is candidate-content-independent)
    cand = germs[:TOPK]
    print(f"segments: {len(segs)} (avg len {sum(len(s) for s in segs)//len(segs)}); "
          f"candidates/read: {TOPK}; projecting to {PAIRS} pairs\n")

    # ---- parasail (proper SW match/mismatch, SIMD) ----
    import parasail
    mat = parasail.matrix_create("ACGTN", 2, -1)
    t = time.perf_counter(); n = 0
    for s in segs:
        for g in cand:
            parasail.sw_striped_16(s, g, 3, 1, mat); n += 1
    dt = time.perf_counter() - t
    pps = n / dt
    print(f"parasail sw_striped_16 : {pps:9.0f} pairs/s -> {PAIRS/pps:6.2f}s for {PAIRS} pairs")

    # ---- edlib (edit distance, infix/HW mode) ----
    import edlib
    t = time.perf_counter(); n = 0
    for s in segs:
        for g in cand:
            edlib.align(s, g, mode="HW", task="distance"); n += 1
    dt = time.perf_counter() - t
    pps = n / dt
    print(f"edlib HW distance      : {pps:9.0f} pairs/s -> {PAIRS/pps:6.2f}s for {PAIRS} pairs")

    # ---- Biopython (current rescore_alleles engine) — subset, it's slow ----
    from Bio import Align
    aln = Align.PairwiseAligner(mode="local")
    aln.match_score, aln.mismatch_score = 1.0, -1.0
    aln.open_gap_score, aln.extend_gap_score = -2.0, -0.5
    sub = segs[:60]
    t = time.perf_counter(); n = 0
    for s in sub:
        for g in cand:
            aln.score(s, g); n += 1
    dt = time.perf_counter() - t
    pps = n / dt
    print(f"Biopython PairwiseAln  : {pps:9.0f} pairs/s -> {PAIRS/pps:6.2f}s for {PAIRS} pairs (extrapolated)")

    print(f"\nreference: GPU soft-DP V rerank ~= 10.4s for 5000 reads (profile_inference.py)")
    print(f"           => ~{10.4*N_READS/5000:.1f}s equivalent for {N_READS} reads")


if __name__ == "__main__":
    main()
