"""`alignair` command-line interface.

    alignair predict reads.fastq -o rearrangement.tsv --model model.pt
    alignair predict reads.fastq -o out.tsv --model model.pt --genotype donor.yaml

A `--genotype` YAML simply becomes the reference for the run, so it transparently supports
both an allele SUBSET and NOVEL alleles (the dynamic-genotype property) with no extra flags.
"""
from __future__ import annotations

import argparse
import json
import os


def cmd_predict(args) -> None:
    import torch
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .inference.dnalignair_infer import predict_reads, canonicalize_sequence
    from .io.sequence_reader import read_sequences
    from .io.airr import write_airr

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location=device)
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    if args.genotype:
        rs = ReferenceSet.from_yaml(args.genotype)
        print(f"reference: genotype {args.genotype} "
              f"(V={len(rs.gene('V').names)}"
              f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    else:
        rs = ReferenceSet.from_dataconfigs(getattr(gdata, args.dataconfig))
        print(f"reference: {args.dataconfig} (V={len(rs.gene('V').names)})")

    calibration = None
    if args.calibration and os.path.exists(args.calibration):
        calibration = json.load(open(args.calibration))

    ids, seqs, info = read_sequences(args.input)
    print(f"read {info['n_read']} sequences ({info['n_dropped']} dropped) as {info['format']}")
    if not seqs:
        raise SystemExit("no valid sequences to align")

    preds = predict_reads(model, rs, seqs, device=device, batch_size=args.batch,
                          rerank="learned", calibration=calibration)
    # coordinates are in the canonical (forward) frame -> emit the canonical sequence so
    # they always match it, even for reverse-complemented input reads (with rev_comp flag).
    canon = [canonicalize_sequence(s, p["orientation_id"]) for s, p in zip(seqs, preds)]
    write_airr(args.output, ids, canon, preds, locus=args.locus)
    print(f"wrote {len(preds)} rearrangements -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alignair",
                                 description="DNAlignAIR — neural IG/TCR sequence aligner")
    sub = ap.add_subparsers(required=True, dest="command")
    pr = sub.add_parser("predict", help="align reads and write an AIRR rearrangement TSV")
    pr.add_argument("input", help="FASTA/FASTQ/CSV/TSV/TXT (optionally .gz) of reads")
    pr.add_argument("-o", "--output", required=True, help="output AIRR rearrangement TSV")
    pr.add_argument("--model", required=True, help="model checkpoint (.pt with {model, config})")
    pr.add_argument("--genotype", default=None,
                    help="genotype YAML used as the reference (allele subset and/or novel alleles)")
    pr.add_argument("--calibration", default=None, help="allele-set calibration JSON")
    pr.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB",
                    help="GenAIRR DataConfig name for the reference when no --genotype")
    pr.add_argument("--locus", default="IGH")
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    pr.set_defaults(func=cmd_predict)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
