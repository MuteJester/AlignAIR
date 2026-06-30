"""Produce the two prediction files for an XAttnAligner-vs-IgBLAST head-to-head on a FROZEN benchmark
case set, keyed by sequence_id=case_id so the benchmark `compare` CLI can pair them. Mirrors
run_h2h_benchmark.py but drives the CURRENT model (XAttnAligner + predict_reads_xattn, incl. the D
inversion-rescue) instead of legacy DNAlignAIR.

Inputs (from `alignair.benchmark.cli build --export-dir ... --export-frame presented`):
  - <export>/<prefix>.fasta            FASTA, header = case_id (presented frame, what aligners see)
  - <export>/<prefix>_airr_input.tsv   sequence_id, sequence (presented)

Outputs:
  - <out>/igblast_airr.tsv             IgBLAST AIRR outfmt-19, sequence_id = case_id
  - <out>/xattn_predictions.jsonl      predict_reads_xattn output + sequence_id = case_id
  - <out>/xattn_performance.json       wall time / throughput sidecar

Then compare:
  PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli compare \
    --cases <cases.jsonl> \
    --a-predictions <out>/igblast_airr.tsv  --a-prediction-format airr-tsv \
    --b-predictions <out>/xattn_predictions.jsonl --b-prediction-format jsonl \
    --model-a-name igblast --model-b-name xattn \
    --policy igh_allele_calling_core --bootstrap 500 --confidence 0.95 \
    --multiple-comparison-correction bonferroni --out <out>/igblast_vs_xattn.json
"""
import argparse
import json
import os
import sys
import time

import torch
sys.path.insert(0, os.path.dirname(__file__))
from run_h2h_benchmark import run_igblast_fasta, load_airr_input  # reuse IgBLAST + IO

import GenAIRR.data as gdata
from alignair.reference.reference_set import ReferenceSet
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.xattn_aligner import XAttnAligner
from alignair.inference.xattn_infer import predict_reads_xattn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True, help="output dir for prediction files")
    ap.add_argument("--model", default=".private/models/xattn_igh.pt")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8, help="IgBLAST threads")
    ap.add_argument("--skip-igblast", action="store_true")
    ap.add_argument("--skip-xattn", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fasta = os.path.join(args.export_dir, f"{args.prefix}.fasta")
    airr_in = os.path.join(args.export_dir, f"{args.prefix}_airr_input.tsv")
    igb_tsv = os.path.join(args.out, "igblast_airr.tsv")
    xa_jsonl = os.path.join(args.out, "xattn_predictions.jsonl")
    xa_perf = os.path.join(args.out, "xattn_performance.json")

    ids, seqs = load_airr_input(airr_in)
    print(f"cases: {len(ids)}  (fasta={fasta})", flush=True)

    # ---- IgBLAST ----
    if not args.skip_igblast:
        print("running IgBLAST ...", flush=True)
        t0 = time.perf_counter()
        run_igblast_fasta(fasta, igb_tsv, threads=args.threads)
        dt = time.perf_counter() - t0
        with open(os.path.join(args.out, "igblast_performance.json"), "w") as fh:
            json.dump({"wall_time_seconds": dt, "n_sequences": len(ids),
                       "reads_per_second": len(ids) / dt, "source": "igblast"}, fh, indent=2)
        print(f"  IgBLAST: {dt:.1f}s  ({len(ids)/dt:.1f} reads/s) -> {igb_tsv}", flush=True)

    # ---- XAttnAligner ----
    if not args.skip_xattn:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading XAttnAligner ({args.model}) on {device} ...", flush=True)
        rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
        ck = torch.load(args.model, map_location="cpu", weights_only=False)
        m = XAttnAligner(DNAlignAIRConfig(**ck["config"]))
        m.load_state_dict(ck["model"]); m.to(device).eval()

        predict_reads_xattn(m, rs, seqs[:min(64, len(seqs))], device=device,
                            batch_size=args.batch_size, topk=args.topk)   # warmup (CUDA init excluded)
        if device == "cuda":
            torch.cuda.synchronize()
        print("running XAttnAligner ...", flush=True)
        t0 = time.perf_counter()
        preds = predict_reads_xattn(m, rs, seqs, device=device, batch_size=args.batch_size, topk=args.topk)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        with open(xa_jsonl, "w") as fh:
            for sid, p in zip(ids, preds):
                p = dict(p); p["sequence_id"] = sid
                fh.write(json.dumps(p) + "\n")
        with open(xa_perf, "w") as fh:
            json.dump({"wall_time_seconds": dt, "n_sequences": len(ids),
                       "reads_per_second": len(ids) / dt,
                       "source": f"xattn:{os.path.basename(args.model)}"}, fh, indent=2)
        print(f"  XAttnAligner: {dt:.1f}s  ({len(ids)/dt:.1f} reads/s) -> {xa_jsonl}", flush=True)


if __name__ == "__main__":
    main()
