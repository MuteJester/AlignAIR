"""Faithful PyTorch AlignAIR vs IgBLAST on the frozen benchmark case set.

Runs predict() on the exported reads, writes the normalized prediction contract + a throughput
sidecar, then prints the `alignair.benchmark.cli compare` command (IgBLAST predictions are already
on disk at <bench>/run/igblast_airr.tsv).
"""
import argparse
import csv
import json
import os
import time

import torch

import GenAIRR.data as gd
from alignair.config.alignair_config import AlignAIRConfig
from alignair.models.single_chain import SingleChainAlignAIR
from alignair.predict import PredictConfig, predict
from alignair.reference.reference_set import ReferenceSet

GENES = ("v", "d", "j")


def to_contract(sid, rec):
    out = {"sequence_id": sid, "sequence": rec["sequence"], "locus": "IGH",
           "orientation_id": rec.get("orientation_id", 0)}
    for g in GENES:
        out[f"{g}_call"] = rec.get(f"{g}_call", "")
        out[f"{g}_calls"] = rec.get(f"{g}_calls", [])
        for f in ("sequence_start", "sequence_end", "germline_start", "germline_end"):
            if f"{g}_{f}" in rec:
                out[f"{g}_{f}"] = rec[f"{g}_{f}"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", default=".private/bench")
    ap.add_argument("--model", default=".private/models/alignair_single_igh.pt")
    ap.add_argument("--out", default=".private/bench/run/alignair_predictions.jsonl")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    a = ap.parse_args()

    ids, seqs = [], []
    with open(os.path.join(a.bench, "igh_airr_input.tsv"), newline="") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            ids.append(row["sequence_id"]); seqs.append(row["sequence"])
    print(f"cases: {len(ids)}  model: {os.path.basename(a.model)}")

    ck = torch.load(a.model, map_location=a.device)
    cfg = AlignAIRConfig(**ck["config"])
    model = SingleChainAlignAIR(cfg).to(a.device).eval()
    model.load_state_dict(ck["model"])
    ref = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    pcfg = PredictConfig(max_seq_length=cfg.max_seq_length, has_d=cfg.has_d, batch_size=a.batch_size)

    predict(model, seqs[:8], ref, pcfg, device=a.device)                 # warmup
    t0 = time.perf_counter()
    records = predict(model, seqs, ref, pcfg, device=a.device)
    dt = time.perf_counter() - t0

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        for sid, rec in zip(ids, records):
            fh.write(json.dumps(to_contract(sid, rec)) + "\n")
    json.dump({"wall_time_seconds": dt, "n_sequences": len(ids),
               "reads_per_second": len(ids) / dt, "source": "alignair"},
              open(a.out.replace(".jsonl", "_performance.json"), "w"), indent=2)
    print(f"alignair: {dt:.1f}s ({len(ids)/dt:.1f} reads/s) -> {a.out}")
    print(f"\ncompare:\n  PYTHONPATH=src .venv/bin/python -m alignair.benchmark.cli compare \\\n"
          f"    --cases {a.bench}/cases.jsonl \\\n"
          f"    --a-predictions {a.bench}/run/igblast_airr.tsv --a-prediction-format airr-tsv \\\n"
          f"    --b-predictions {a.out} --b-prediction-format jsonl \\\n"
          f"    --model-a-name igblast --model-b-name alignair \\\n"
          f"    --policy igh_allele_calling_core --bootstrap 500 --confidence 0.95 \\\n"
          f"    --multiple-comparison-correction bonferroni --out {a.bench}/run/igblast_vs_alignair.json")


if __name__ == "__main__":
    main()
