"""Run a DNAlignAIR checkpoint over a benchmark case JSONL and emit prediction JSONL
(matched by sequence_id=case_id, canonical frame) for `alignair.benchmark.cli evaluate`.
"""
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
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--cases", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--frame", choices=("presented", "canonical"), default="canonical")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    ckpt = torch.load(args.model, map_location=device)
    cfg = DNAlignAIRConfig(**ckpt["config"]) if isinstance(ckpt["config"], dict) else ckpt["config"]
    model = DNAlignAIR(cfg).to(device); model.load_state_dict(ckpt["model"]); model.eval()
    calibration = None
    if args.calibration and os.path.exists(args.calibration):
        calibration = json.load(open(args.calibration))
        print(f"loaded calibration: { {g: round(c['epsilon'],2) for g,c in calibration.items()} }")

    cases = [json.loads(l) for l in open(args.cases)]
    ids = [c["case_id"] for c in cases]
    frame = args.frame
    # feed the PRESENTED read so the model's orientation head is exercised for real;
    # predict_reads canonicalizes internally and returns canonical-frame coords.
    seqs = [c["sequence"] if frame == "presented" else c["canonical_sequence"] for c in cases]
    print(f"scoring {len(seqs)} cases ({frame} input) with {args.model} ...")

    preds = predict_reads(model, rs, seqs, device=device, batch_size=args.batch,
                          topk=32, rerank="learned", calibration=calibration)
    with open(args.out, "w") as f:
        for cid, p in zip(ids, preds):
            row = {"sequence_id": cid,
                   "orientation_id": p["orientation_id"],
                   "productive": p["productive"],
                   "mutation_rate": p["mutation_rate"],
                   "indel_count": p["indel_count"]}
            for g in ("v", "d", "j"):
                row[f"{g}_call"] = p[f"{g}_call"]
                row[f"{g}_calls"] = p.get(f"{g}_call_set") or [p[f"{g}_call"]]
                row[f"{g}_ranked_calls"] = p.get(f"{g}_topk", [])
                row[f"{g}_resolved_call"] = p.get(f"{g}_resolved_call")
                row[f"{g}_call_level"] = p.get(f"{g}_call_level")
                for k in ("sequence_start", "sequence_end", "germline_start", "germline_end"):
                    row[f"{g}_{k}"] = p[f"{g}_{k}"]
            f.write(json.dumps(row) + "\n")
    print(f"wrote predictions -> {args.out}")


if __name__ == "__main__":
    main()
