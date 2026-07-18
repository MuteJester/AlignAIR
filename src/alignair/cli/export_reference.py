"""`alignair export-reference` — dump the embedded germline FASTA or a dataconfig pickle."""
from __future__ import annotations

from ..model_file import read_dataconfig, read_reference


def register(sub) -> None:
    p = sub.add_parser("export-reference", help="export the germline FASTA / dataconfig from a model file")
    p.add_argument("model")
    p.add_argument("--fasta", help="write the germline V/D/J FASTA here")
    p.add_argument("--dataconfig", help="write the pickled GenAIRR DataConfig here")
    p.add_argument("--index", type=int, default=0, help="which embedded dataconfig (multi-chain)")
    p.set_defaults(func=run)


def run(args) -> int:
    if not args.fasta and not args.dataconfig:
        print("nothing to export: pass --fasta and/or --dataconfig")
        return 1
    if args.fasta:
        with open(args.fasta, "w") as f:
            f.write(read_reference(args.model))
        print(f"wrote {args.fasta}")
    if args.dataconfig:
        import pickle
        with open(args.dataconfig, "wb") as f:
            pickle.dump(read_dataconfig(args.model, index=args.index), f, protocol=5)
        print(f"wrote {args.dataconfig}")
    return 0
