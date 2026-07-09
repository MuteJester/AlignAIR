"""``alignair predict`` — align reads with a trained model and write an AIRR TSV."""
from __future__ import annotations

from ..api import load_model, predict_sequences
from ..io.airr import write_airr
from ..io.sequence_reader import read_sequences


def register(sub) -> None:
    p = sub.add_parser("predict", help="align reads with a trained model -> AIRR TSV")
    p.add_argument("--model", required=True, help="path to a trained AlignAIR checkpoint (.pt)")
    p.add_argument("--dataconfig", nargs="+", required=True,
                   help="GenAIRR dataconfig(s) for the germline reference (must match the model)")
    p.add_argument("--input", required=True, help="reads to align (FASTA / FASTQ / TSV; .gz ok; '-' = stdin)")
    p.add_argument("--out", required=True, help="output AIRR TSV path")
    p.add_argument("--locus", default="IGH", help="locus label for the AIRR output")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.add_argument("--batch-size", type=int, default=64)
    p.set_defaults(func=run)


def run(args) -> int:
    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    ids, seqs, stats = read_sequences(args.input)
    if not seqs:
        print(f"no valid reads in {args.input}")
        return 1
    model, reference = load_model(args.model, dataconfigs=args.dataconfig, device=device)
    records = predict_sequences(model, reference, seqs, device=device, batch_size=args.batch_size)
    write_airr(args.out, ids, seqs, records, locus=args.locus)
    print(f"aligned {len(records)} reads ({stats['n_dropped']} dropped) -> {args.out}")
    return 0
