"""`alignair` command-line interface.

    alignair predict reads.fastq -o rearrangement.tsv --model model.pt
    alignair predict reads.fastq -o out.tsv --model model.pt --genotype donor.yaml
    alignair predict reads.fastq -o out.tsv --model model.pt --genotype donor.fasta

A `--genotype` file (YAML or FASTA) simply becomes the reference for the run, so it
transparently supports both an allele SUBSET and NOVEL alleles (the dynamic-genotype
property) with no extra flags.
"""
from __future__ import annotations

import argparse
import json
import os
import sys


def _version() -> str:
    try:
        from importlib.metadata import version
        return version("AlignAIR")
    except Exception:
        return "unknown (not pip-installed)"


def _load_model(model_path, device):
    """Load a DNAlignAIR from either a versioned bundle directory or a raw {model, config}
    .pt checkpoint. Returns (model, dataconfigs, locus, calibration_or_None)."""
    import torch
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .serialization.dnalignair_bundle import is_bundle, load_dnalignair_bundle
    if is_bundle(model_path):
        b = load_dnalignair_bundle(model_path, build=True, device=device)
        return b["model"], b["dataconfigs"], b["locus"], b["calibration"]
    ckpt = torch.load(model_path, map_location=device)
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, None, None, None


def cmd_predict(args) -> None:
    import torch
    import GenAIRR.data as gdata
    from .reference.reference_set import ReferenceSet
    from .inference.dnalignair_infer import predict_reads, canonicalize_sequence
    from .io.sequence_reader import read_sequences
    from .io.airr import write_airr

    def log(msg):
        if not args.quiet:
            print(msg, flush=True)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    if not os.path.exists(args.model):
        raise SystemExit(f"error: model not found: {args.model}")
    model, b_dataconfigs, b_locus, b_calibration = _load_model(args.model, device)
    locus = args.locus or b_locus or "IGH"
    log(f"device: {device}  |  model: {args.model}")

    if args.genotype:
        if not os.path.exists(args.genotype):
            raise SystemExit(f"error: genotype file not found: {args.genotype}")
        ext = os.path.splitext(args.genotype)[1].lower()
        loader = ReferenceSet.from_fasta if ext in (".fasta", ".fa", ".fna", ".faa") else ReferenceSet.from_yaml
        rs = loader(args.genotype)
        log(f"reference: genotype {args.genotype} "
            f"(V={len(rs.gene('V').names)}"
            f"{', D=' + str(len(rs.gene('D').names)) if rs.has_d else ''}, J={len(rs.gene('J').names)})")
    else:
        names = b_dataconfigs or [args.dataconfig]
        try:
            rs = ReferenceSet.from_dataconfigs(*[getattr(gdata, n) for n in names])
        except AttributeError as e:
            raise SystemExit(f"error: unknown GenAIRR DataConfig in {names}: {e}")
        log(f"reference: {', '.join(names)} (V={len(rs.gene('V').names)})")

    calibration = b_calibration
    if args.calibration and os.path.exists(args.calibration):
        calibration = json.load(open(args.calibration))           # explicit flag overrides bundle

    ids, seqs, info = read_sequences(args.input)
    log(f"read {info['n_read']} sequences ({info['n_dropped']} dropped) as {info['format']}")
    if not seqs:
        raise SystemExit("error: no valid sequences to align")

    preds = predict_reads(model, rs, seqs, device=device, batch_size=args.batch,
                          rerank="learned", v_reader=args.v_reader, calibration=calibration)
    # coordinates are in the canonical (forward) frame -> emit the canonical sequence so
    # they always match it, even for reverse-complemented input reads (with rev_comp flag).
    canon = [canonicalize_sequence(s, p["orientation_id"]) for s, p in zip(seqs, preds)]
    out_dir = os.path.dirname(os.path.abspath(args.output))
    try:
        os.makedirs(out_dir, exist_ok=True)
        write_airr(args.output, ids, canon, preds, locus=locus)
    except (PermissionError, OSError) as e:
        raise SystemExit(
            f"error: cannot write output to {args.output}: {e}\n"
            f"hint: if running in Docker, mount a writable output directory and add "
            f"`--user $(id -u):$(id -g)` so files are written as you.")
    log(f"wrote {len(preds)} rearrangements -> {args.output}")


def cmd_doctor(args) -> None:
    """Environment / install check: Python, PyTorch + CUDA, GenAIRR, optional parasail, and
    (optionally) whether a --model path resolves. Exit non-zero if a CORE dependency is missing."""
    ok = True
    print(f"AlignAIR {_version()}")
    print(f"  python      : {sys.version.split()[0]} ({sys.executable})")
    try:
        import torch
        cuda = torch.cuda.is_available()
        dev = torch.cuda.get_device_name(0) if cuda else "cpu only"
        print(f"  torch       : {torch.__version__}  | CUDA available: {cuda} ({dev})")
    except Exception as e:
        ok = False; print(f"  torch       : MISSING ({e})")
    try:
        import GenAIRR
        print(f"  GenAIRR     : {getattr(GenAIRR, '__version__', '?')}")
    except Exception as e:
        ok = False; print(f"  GenAIRR     : MISSING ({e})")
    try:
        import parasail  # noqa: F401
        print("  parasail    : present (fast V reader available via --v-reader parasail)")
    except Exception:
        print("  parasail    : absent (optional; install AlignAIR[reader] for the fast V reader)")
    if args.model:
        from .serialization.dnalignair_bundle import is_bundle
        if not os.path.exists(args.model):
            ok = False; print(f"  model       : NOT FOUND ({args.model})")
        else:
            print(f"  model       : {'bundle' if is_bundle(args.model) else 'raw checkpoint'} at {args.model}")
    print("status: OK" if ok else "status: PROBLEMS FOUND")
    raise SystemExit(0 if ok else 1)


def cmd_bundle(args) -> None:
    """Package a raw {model, config} checkpoint (+ optional calibration) into a versioned,
    fingerprinted DNAlignAIR bundle directory."""
    import torch
    from .config.dnalignair_config import DNAlignAIRConfig
    from .core.dnalignair import DNAlignAIR
    from .serialization.dnalignair_bundle import save_dnalignair_bundle
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = ckpt["config"]
    cfg = DNAlignAIRConfig(**cfg) if isinstance(cfg, dict) else cfg
    model = DNAlignAIR(cfg)
    model.load_state_dict(ckpt["model"])
    calibration = json.load(open(args.calibration)) if args.calibration else None
    dataconfigs = args.dataconfig or ["HUMAN_IGH_OGRDB"]
    path = save_dnalignair_bundle(args.output, model=model, dataconfigs=dataconfigs,
                                  locus=args.locus or "IGH", calibration=calibration, notes=args.notes)
    print(f"wrote DNAlignAIR bundle -> {path}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="alignair",
                                 description="AlignAIR — neural IG/TCR sequence aligner")
    ap.add_argument("--version", action="version", version=f"AlignAIR {_version()}")
    sub = ap.add_subparsers(required=True, dest="command")
    pr = sub.add_parser("predict", help="align reads and write an AIRR rearrangement TSV")
    pr.add_argument("input", help="FASTA/FASTQ/CSV/TSV/TXT (optionally .gz) of reads")
    pr.add_argument("-o", "--output", required=True, help="output AIRR rearrangement TSV")
    pr.add_argument("--model", required=True,
                    help="a DNAlignAIR bundle directory OR a raw .pt checkpoint {model, config}")
    pr.add_argument("--genotype", default=None,
                    help="genotype file (YAML or FASTA) used as the reference "
                         "(allele subset and/or novel alleles)")
    pr.add_argument("--calibration", default=None,
                    help="allele-set calibration JSON (overrides a bundled calibration)")
    pr.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB",
                    help="GenAIRR DataConfig name for the reference (raw checkpoint, no --genotype)")
    pr.add_argument("--locus", default=None, help="locus label (default: bundle's, else IGH)")
    pr.add_argument("--batch", type=int, default=64)
    pr.add_argument("--device", default=None, help="cuda|cpu (auto if unset)")
    pr.add_argument("--v-reader", default="learned", choices=["learned", "parasail"],
                    help="V allele reader: learned (default) or parasail (faster+sharper; needs AlignAIR[reader])")
    pr.add_argument("--quiet", action="store_true", help="suppress progress output")
    pr.set_defaults(func=cmd_predict)

    dr = sub.add_parser("doctor", help="check the environment (Python, torch+CUDA, GenAIRR, parasail)")
    dr.add_argument("--model", default=None, help="optionally verify a model bundle/checkpoint resolves")
    dr.set_defaults(func=cmd_doctor)

    bd = sub.add_parser("bundle", help="package a raw checkpoint into a versioned bundle")
    bd.add_argument("--model", required=True, help="raw .pt checkpoint {model, config}")
    bd.add_argument("-o", "--output", required=True, help="bundle directory to create")
    bd.add_argument("--calibration", default=None, help="allele-set calibration JSON to include")
    bd.add_argument("--dataconfig", action="append", default=None,
                    help="GenAIRR DataConfig name(s) for the default reference (repeatable)")
    bd.add_argument("--locus", default=None)
    bd.add_argument("--notes", default=None)
    bd.set_defaults(func=cmd_bundle)
    return ap


def main(argv=None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
