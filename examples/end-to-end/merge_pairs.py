#!/usr/bin/env python3
"""Minimal, zero-dependency paired-read merger for the worked example.

Merges R1 with the reverse complement of R2 at the largest overlap that has <=10% mismatch.
This is a demo-grade merger with no external dependencies so the worked example runs anywhere
Python does; it is deterministic, so the merged output has a stable checksum. For production
repertoire work use a proper tool (pRESTO AssemblePairs, fastp --merge, or vsearch), ideally with
UMI-based consensus.

Usage:
    python3 merge_pairs.py R1.fastq R2.fastq merged.fastq
"""
import sys


def rc(s):
    c = {"A": "T", "T": "A", "G": "C", "C": "G", "N": "N"}
    return "".join(c.get(b, "N") for b in reversed(s))


def reads(path):
    with open(path) as f:
        while True:
            h = f.readline()
            if not h:
                break
            s = f.readline().strip()
            f.readline()
            f.readline()
            yield h.strip().split()[0][1:], s


def merge(s1, s2rc, minov=20, maxmm=0.10):
    L1 = len(s1)
    for ov in range(min(L1, len(s2rc)), minov - 1, -1):
        a, b = s1[L1 - ov:], s2rc[:ov]
        mm = sum(1 for x, y in zip(a, b) if x != y and "N" not in (x, y))
        if mm <= max(1, int(maxmm * ov)):
            return s1 + s2rc[ov:]
    return None


def main():
    r1, r2, out = sys.argv[1], sys.argv[2], sys.argv[3]
    pairs = merged = 0
    with open(out, "w") as w:
        for (i1, s1), (_i2, s2) in zip(reads(r1), reads(r2)):
            pairs += 1
            frag = merge(s1, rc(s2))
            if frag:
                merged += 1
                w.write(f"@{i1}\n{frag}\n+\n{'I' * len(frag)}\n")
    sys.stderr.write(f"pairs={pairs} merged={merged}\n")


if __name__ == "__main__":
    main()
