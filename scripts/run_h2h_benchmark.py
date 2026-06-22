"""Produce the two prediction files for a DNAlignAIR-vs-IgBLAST head-to-head on a
FROZEN benchmark case set, keyed by sequence_id=case_id so the benchmark `compare`
CLI can pair them.

Inputs (from `alignair.benchmark.cli build --export-dir ...`):
  - <export>/<prefix>.fasta            FASTA, header = case_id (what aligners see, presented frame)
  - <export>/<prefix>_airr_input.tsv   sequence_id, sequence (presented)

Outputs:
  - <out>/igblast_airr.tsv             IgBLAST AIRR outfmt-19, sequence_id = case_id
  - <out>/dnalignair_predictions.jsonl predict_reads output + sequence_id = case_id
  - <out>/dnalignair_performance.json  wall time / throughput sidecar for DNAlignAIR

Then run the comparison with:
  PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli compare \
    --cases <cases.jsonl> \
    --a-predictions <out>/igblast_airr.tsv  --a-prediction-format airr-tsv \
    --b-predictions <out>/dnalignair_predictions.jsonl --b-prediction-format jsonl \
    --model-a-name igblast --model-b-name dnalignair \
    --policy igh_allele_calling_core --bootstrap 500 --confidence 0.95 \
    --multiple-comparison-correction bonferroni --out <out>/igblast_vs_dnalignair.json
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(__file__))
from baseline_igblast import IGB, TOOLS  # noqa: E402  (reuse igblast install paths)

import GenAIRR.data as gdata  # noqa: E402
from alignair.reference.reference_set import ReferenceSet  # noqa: E402
from alignair.config.dnalignair_config import DNAlignAIRConfig  # noqa: E402
from alignair.core.dnalignair import DNAlignAIR  # noqa: E402
from alignair.inference.dnalignair_infer import predict_reads  # noqa: E402


def run_igblast_fasta(fasta_path: str, out_tsv: str, threads: int = 8) -> None:
    """Run igblastn (AIRR outfmt 19) directly on a FASTA whose headers are case_ids,
    so the AIRR sequence_id column == case_id. Writes out_tsv in place."""
    env = dict(os.environ, IGDATA=IGB)
    cmd = [os.path.join(IGB, "bin", "igblastn"),
           "-germline_db_V", os.path.join(TOOLS, "germline", "igh_v"),
           "-germline_db_D", os.path.join(TOOLS, "germline", "igh_d"),
           "-germline_db_J", os.path.join(TOOLS, "germline", "igh_j"),
           "-auxiliary_data", os.path.join(IGB, "optional_file", "human_gl.aux"),
           "-organism", "human", "-ig_seqtype", "Ig", "-num_threads", str(threads),
           "-query", fasta_path, "-outfmt", "19", "-out", out_tsv]
    subprocess.run(cmd, env=env, check=True, capture_output=True)


def load_airr_input(tsv_path: str):
    """Return (ids, sequences) in file order from the exported AIRR input table."""
    ids, seqs = [], []
    with open(tsv_path, newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ids.append(row["sequence_id"])
            seqs.append(row["sequence"])
    return ids, seqs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--export-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--out", required=True, help="output dir for prediction files")
    ap.add_argument("--model", default=".private/models/scaled_long.pt")
    ap.add_argument("--calibration", default=".private/models/allele_set_calibration.json")
    ap.add_argument("--rerank", default="learned", choices=["learned", "none"])
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--topk", type=int, default=16)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-igblast", action="store_true")
    ap.add_argument("--skip-dnalignair", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    fasta = os.path.join(args.export_dir, f"{args.prefix}.fasta")
    airr_in = os.path.join(args.export_dir, f"{args.prefix}_airr_input.tsv")
    igb_tsv = os.path.join(args.out, "igblast_airr.tsv")
    da_jsonl = os.path.join(args.out, "dnalignair_predictions.jsonl")
    da_perf = os.path.join(args.out, "dnalignair_performance.json")

    ids, seqs = load_airr_input(airr_in)
    print(f"cases: {len(ids)}  (fasta={fasta})")

    # ---- IgBLAST ----
    if not args.skip_igblast:
        print("running IgBLAST ...")
        t0 = time.perf_counter()
        run_igblast_fasta(fasta, igb_tsv, threads=args.threads)
        dt = time.perf_counter() - t0
        with open(os.path.join(args.out, "igblast_performance.json"), "w") as fh:
            json.dump({"wall_time_seconds": dt, "n_sequences": len(ids),
                       "reads_per_second": len(ids) / dt, "source": "igblast"}, fh, indent=2)
        print(f"  IgBLAST: {dt:.1f}s  ({len(ids)/dt:.1f} reads/s) -> {igb_tsv}")

    # ---- DNAlignAIR ----
    if not args.skip_dnalignair:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"loading DNAlignAIR ({args.model}) on {device} ...")
        rs = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
        ck = torch.load(args.model, map_location=device)
        cfg = DNAlignAIRConfig(**ck["config"])
        m = DNAlignAIR(cfg).to(device); m.load_state_dict(ck["model"]); m.eval()
        cal = json.load(open(args.calibration)) if os.path.exists(args.calibration) else None

        # warmup (CUDA init excluded from the timed run)
        with torch.no_grad():
            predict_reads(m, rs, seqs[:min(64, len(seqs))], device=device,
                          topk=args.topk, rerank=args.rerank, calibration=cal)
        if device == "cuda":
            torch.cuda.synchronize()
        print("running DNAlignAIR ...")
        t0 = time.perf_counter()
        with torch.no_grad():
            preds = predict_reads(m, rs, seqs, device=device, batch_size=args.batch_size,
                                  topk=args.topk, rerank=args.rerank, calibration=cal)
        if device == "cuda":
            torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        with open(da_jsonl, "w") as fh:
            for sid, p in zip(ids, preds):
                p = dict(p); p["sequence_id"] = sid
                fh.write(json.dumps(p) + "\n")
        with open(da_perf, "w") as fh:
            json.dump({"wall_time_seconds": dt, "n_sequences": len(ids),
                       "reads_per_second": len(ids) / dt,
                       "source": f"dnalignair:{os.path.basename(args.model)}:rerank={args.rerank}"},
                      fh, indent=2)
        print(f"  DNAlignAIR: {dt:.1f}s  ({len(ids)/dt:.1f} reads/s) -> {da_jsonl}")

    print("\ndone. now run `alignair.benchmark.cli compare` (see header docstring).")


if __name__ == "__main__":
    main()
