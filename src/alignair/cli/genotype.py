"""``alignair genotype`` — infer an individual's IG genotype from a repertoire (experimental).

Writes an explainable report (`<stem>.genotype.report.{txt,json}`) and an AIRR GenotypeSet
(`<stem>.genotype.airr.json`). Separate from `predict` so a genotype run is never confused with an
alignment run.
"""
from __future__ import annotations

import json


def register(sub) -> None:
    p = sub.add_parser("genotype", help="infer an individual's IG genotype from a repertoire (experimental)")
    p.add_argument("input", help="repertoire reads (FASTA / FASTQ / TSV; .gz ok)")
    p.add_argument("--model", required=True, help="a .alignair/.pt path or a shipped registry id")
    p.add_argument("--dataconfig", nargs="+", default=None)
    p.add_argument("--out", default=None, help="output stem (default: from --input)")
    p.add_argument("--locus", default="IGH")
    p.add_argument("--min-support", type=float, default=0.003, help="min per-allele support fraction")
    p.add_argument("--germline-set-ref", default=None, help="CURIE for the source GermlineSet (AIRR)")
    p.add_argument("--device", default=None)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--trust-pickle", action="store_true")
    p.add_argument("--registry", action="append")
    p.add_argument("--offline", action="store_true")
    p.set_defaults(func=run)


def run(args) -> int:
    import os

    import torch
    from ..api import load_model
    from ..genotype import infer_genotype
    from ..genotype.report import format_report
    from ..io.sequence_reader import read_sequences
    from ..registry import resolve_model, sources as _sources
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    _, seqs, stats = read_sequences(args.input)
    if not seqs:
        print(f"no valid reads in {args.input}")
        return 1
    try:
        model_path = str(resolve_model(args.model, sources=_sources.resolve_sources(args.registry),
                                       offline=args.offline))
    except Exception as e:
        print(f"could not resolve model '{args.model}': {e}")
        return 1
    try:
        model, reference = load_model(model_path, dataconfigs=args.dataconfig, device=device,
                                      trust_pickle=args.trust_pickle)
    except ValueError as e:
        print(str(e))
        return 1

    result = infer_genotype(model, reference, seqs, locus=args.locus, min_support=args.min_support,
                            germline_set_ref=args.germline_set_ref, device=device, batch_size=args.batch_size)
    stem = args.out or (args.input.rsplit(".", 1)[0])
    with open(f"{stem}.genotype.report.json", "w") as f:
        json.dump(result.report, f, indent=2)
    with open(f"{stem}.genotype.report.txt", "w") as f:
        f.write(format_report(result.report) + "\n")
    with open(f"{stem}.genotype.airr.json", "w") as f:
        json.dump(result.genotype_set, f, indent=2)
    for w in result.warnings:
        print(f"warning: {w}")
    print(f"genotyped {len(seqs)} reads -> {stem}.genotype.{{report.txt,report.json,airr.json}}")
    return 0
