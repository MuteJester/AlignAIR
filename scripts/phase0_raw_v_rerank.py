"""Phase-0 accuracy A/B: re-rank the V call by RAW-nucleotide parasail SW over the model's
own top-k candidates (at the predicted V segment), recompute junction, write a new prediction
JSONL. No model rerun, no retrain — post-processes an existing dnalignair_predictions.jsonl.

This tests the architectural bet: does a classical raw-base reader over neural top-k beat the
learned soft-DP reader on the proper fixture, without regressing anything?"""
import argparse, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
import parasail
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.inference.dnalignair_infer import junction_fields

MAT = parasail.matrix_create("ACGTN", 2, -1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default="experiments/headtohead/preds/dnalignair_predictions.jsonl")
    ap.add_argument("--out", default="experiments/headtohead/preds/dnalignair_rawv_predictions.jsonl")
    args = ap.parse_args()

    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    vref = rs.gene("V")
    germ = dict(zip(vref.names, vref.sequences))

    n = changed = 0
    with open(args.out, "w") as out:
        for l in open(args.inp):
            p = json.loads(l); n += 1
            seq = p.get("sequence"); topk = p.get("v_topk") or []
            vs, ve = p.get("v_sequence_start"), p.get("v_sequence_end")
            if seq and topk and vs is not None and ve and ve - vs >= 5:
                seg = seq[vs:ve].upper()
                best, best_s = p.get("v_call"), float("-inf")
                for c in topk:
                    g = germ.get(c)
                    if not g:
                        continue
                    s = parasail.sw_striped_16(seg, g, 3, 1, MAT).score
                    if s > best_s:
                        best_s, best = s, c
                if best != p.get("v_call"):
                    changed += 1
                    p["v_call"] = best
                    p["v_calls"] = [best]                     # collapse set to the raw call for scoring
                    # recompute junction with the (possibly) new V anchor
                    p.update(junction_fields(p, seq, rs))
            out.write(json.dumps(p) + "\n")
    print(f"reranked {n} preds; V call changed on {changed} ({changed/n*100:.1f}%) -> {args.out}")


if __name__ == "__main__":
    main()
