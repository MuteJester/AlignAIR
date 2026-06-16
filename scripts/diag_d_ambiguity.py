"""Data-only diagnostic: is D's low top-1-in-set accuracy real error or irreducible
ambiguity? Streams IGH records at the HARDEST curriculum setting and reports, per gene,
the multi-allele degree of the ground truth and (for D) how much D signal survives."""
import argparse
import numpy as np

import GenAIRR.data as gdata
from alignair.gym.curriculum import Curriculum
from alignair.gym.gym import build_experiment


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="HUMAN_IGH_OGRDB")
    ap.add_argument("--n", type=int, default=3000)
    ap.add_argument("--p", type=float, default=1.0, help="curriculum progress (1.0 = hardest)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    dc = getattr(gdata, args.config)
    params = Curriculum().params(args.p)
    exp = build_experiment(dc, params)
    print(f"config={args.config} p={args.p} | corruption={params}")

    genes = ("v", "d", "j")
    ndeg = {g: [] for g in genes}        # number of GT alleles per call
    dlen = []                            # D in-sequence span length
    dpresent = 0
    total = 0
    for rec in exp.stream_records(n=args.n, seed=args.seed):
        total += 1
        for g in genes:
            call = rec.get(f"{g}_call")
            if call:
                ndeg[g].append(len(str(call).split(",")))
        ds, de = rec.get("d_sequence_start"), rec.get("d_sequence_end")
        if ds is not None and de is not None:
            span = int(de) - int(ds)
            dlen.append(span)
            if span > 0:
                dpresent += 1

    print(f"\nstreamed {total} records")
    for g in genes:
        a = np.array(ndeg[g]) if ndeg[g] else np.array([0])
        print(f"  {g.upper()}: present={len(ndeg[g])}/{total} "
              f"| mean_alleles={a.mean():.2f} multi(%)={100*np.mean(a>1):.1f} "
              f"| max={a.max()}")

    if dlen:
        d = np.array(dlen)
        print(f"\nD in-sequence span (bp): mean={d.mean():.1f} median={np.median(d):.0f} "
              f"min={d.min()} max={d.max()}")
        for thr in (0, 3, 5, 8, 11):
            print(f"  D span <= {thr:2d} bp: {100*np.mean(d <= thr):.1f}%")
        print(f"  D present (span>0): {100*dpresent/total:.1f}%")


if __name__ == "__main__":
    main()
