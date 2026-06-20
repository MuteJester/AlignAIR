"""Measure the value of hierarchical degradation + abstention on fragments: instead of a
forced (often wrong) allele top1, how often do we give a CORRECT coarser call (gene/family)
or HONESTLY abstain — vs make a hard error?"""
import argparse
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=".private/models/scaled_novel.pt")
    ap.add_argument("--cases", default="experiments/igh_bench.jsonl")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

    cases = [json.loads(l) for l in open(args.cases)]
    for frag in ("fragment_120", "fragment_80", "fragment_50"):
        rows = [c for c in cases if frag in c["case_id"]]
        seqs = [c["canonical_sequence"] for c in rows]
        preds = predict_reads(model, rs, seqs, device=device, rerank="learned", calibration=cal)
        n = len(rows)
        flat_hit = 0          # forced allele top1 correct
        correct = {"allele": 0, "gene": 0, "family": 0}
        abstain = 0
        error = 0             # gave a non-abstain call that was WRONG at its level
        for c, p in zip(rows, preds):
            truth = set(c["genes"]["v"]["calls"])
            tg = {a.split("*")[0] for a in truth}
            tf = {a.split("-")[0] for a in truth}
            flat_hit += int(p["v_call"] in truth)
            lvl = p["v_call_level"]; rc = p["v_resolved_call"]
            if lvl == "none" or rc is None:
                abstain += 1
            elif lvl == "allele":
                if rc in truth: correct["allele"] += 1
                else: error += 1
            elif lvl == "gene":
                if rc in tg: correct["gene"] += 1
                else: error += 1
            elif lvl == "family":
                if rc in tf: correct["family"] += 1
                else: error += 1
        useful = sum(correct.values())
        print(f"\n[{frag}] n={n}")
        print(f"  flat allele top1 (forced):      acc={flat_hit/n:.2f}  (error={1-flat_hit/n:.2f})")
        print(f"  hierarchical: correct allele={correct['allele']/n:.2f} gene={correct['gene']/n:.2f} "
              f"family={correct['family']/n:.2f}")
        print(f"     -> useful correct call={useful/n:.2f}  honest abstain={abstain/n:.2f}  "
              f"hard error={error/n:.2f}")


if __name__ == "__main__":
    main()
