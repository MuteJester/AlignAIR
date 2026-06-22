"""Calibrate the parasail V-reader equivalence-set band (raw_set_band) on a HELD-OUT sample
(seed != fixture) by maximizing mean call_set_f1 vs truth. The point call (v_calls[0]) is band-
independent; this only sizes the set. Mirrors the learned path's epsilon calibration but on raw
parasail scores."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import json, torch
from baseline_igblast import gen_records
import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.inference.dnalignair_infer import predict_reads


def f1_set(pred_set, truth_set):
    p, t = set(pred_set), set(truth_set)
    if not p or not t:
        return 0.0
    inter = len(p & t)
    if inter == 0:
        return 0.0
    rec, prec = inter / len(t), inter / len(p)
    return 2 * rec * prec / (rec + prec)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ck = torch.load(".private/models/scaled_long.pt", map_location=device)
    cfg = DNAlignAIRConfig(**ck["config"]); m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
    cal = json.load(open(".private/models/allele_set_calibration.json"))

    # held-out: hard full reads (seed 999) + heavy-SHM (where sets matter) + some fragments
    recs = (gen_records(1.0, 800, 999, None)
            + gen_records(1.0, 400, 999, None, overrides={"mutation_rate": 0.22})
            + gen_records(1.0, 300, 999, 120))
    reads = [r["sequence"] for r in recs]
    preds = predict_reads(m, rs, reads, device=device, topk=16, rerank="learned",
                          calibration=cal, v_reader="parasail", emit_scores=True)

    bands = [2, 3, 4, 5, 6, 8, 10, 14, 20]
    print(f"n={len(recs)} held-out reads\n{'band':>5} {'set_f1':>8} {'avg_size':>9} {'top1_acc':>9}")
    best = (None, -1)
    for band in bands:
        f1s, sizes, top1 = [], [], []
        for rec, p in zip(recs, preds):
            truth = rec.get("v_call")
            if not truth:
                continue
            tset = set(str(truth).split(","))
            sc = p.get("v_scores") or []
            if not sc:
                continue
            top = max(s for _, s in sc)
            keep = sorted([(nm, s) for nm, s in sc if top - s <= band], key=lambda x: -x[1])
            pset = [nm for nm, _ in keep]
            f1s.append(f1_set(pset, tset)); sizes.append(len(pset))
            top1.append(1.0 if (pset[0] if pset else None) in tset else 0.0)
        mf1 = sum(f1s) / len(f1s)
        print(f"{band:>5} {mf1:8.3f} {sum(sizes)/len(sizes):9.2f} {sum(top1)/len(top1):9.3f}")
        if mf1 > best[1]:
            best = (band, mf1)
    print(f"\nBEST band = {best[0]} (set_f1 ={best[1]:.3f})")


if __name__ == "__main__":
    main()
