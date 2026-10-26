"""``alignair predict-aa`` — the amino-acid inference product CLI verb (design spec §10).

Reads a FASTA file (or ``-`` for stdin), runs the AA product path (§6/§7/§9), and writes one JSON
object per line to ``--out`` plus a ``<out>.run.json`` provenance sidecar. A separate verb from the
nucleotide ``alignair predict`` — different model family, different result schema (no CIGAR, no
productivity, no nucleotide coordinates); see ``alignair.aa.product.schema.AAPrediction``.

**File parsing lives HERE** (§10), not in the public façade — ``alignair.predict_aa`` accepts only
in-memory sequences/records, never a path (§9). This module deliberately does **not** reuse
``alignair.io.sequence_reader.read_sequences``/``iter_sequences``: their shared content validator
(``validate_sequence``) cleans a *nucleotide* alphabet (``ACGTN`` + IUPAC ambiguity codes -> ``N``),
which would corrupt or outright drop real amino-acid sequences — almost every AA letter outside
``A``/``C``/``G``/``T`` falls outside that alphabet and gets rewritten to ``N``, which for a real
protein sequence trips the reader's own >20%-ambiguous rejection and silently discards the record.
AA-specific validation (20-residue alphabet, ``X``, no ``*``, no interior whitespace, overlength) is
applied exactly once, downstream, inside ``AAPredictor.predict`` itself
(``alignair.aa.product.validate.normalize_inputs``) — this module only turns file bytes into raw
``{sequence_id, sequence}`` records, verbatim.
"""
from __future__ import annotations

import gzip
import json
import sys
from datetime import datetime, timezone

from .loader import load_aa_model
from .predictor import _provenance
from .validate import ErrorResult


def register(sub) -> None:
    p = sub.add_parser("predict-aa", help="amino-acid inference product: FASTA -> §6 JSONL predictions")
    p.add_argument("input", help="reads to predict (FASTA, or plain text one sequence per line; "
                                 "'-' = stdin; '.gz' ok)")
    p.add_argument("--model", default="alignair-aa-igh-human@2.0.0",
                   help="AA model id or .alignair path (default: alignair-aa-igh-human@2.0.0)")
    p.add_argument("--coverage-group", choices=["full", "trunc", "short"], default=None,
                   help="§7 Mondrian calibration group to apply to every record "
                        "(default: pooled fallback -- see §7)")
    p.add_argument("--on-invalid", choices=["error", "skip"], default="skip",
                   help="error = fail fast on invalid sequence; skip = emit per-record error (default: skip)")
    p.add_argument("--device", default="auto", help="cpu / cuda / auto (default: auto)")
    p.add_argument("--out", required=True, help="output path: one §6 AAPrediction JSON object per line")
    p.add_argument("--no-run-metadata", action="store_true",
                   help="do not write the <out>.run.json provenance sidecar")
    p.set_defaults(func=run)


def _read_text(path: str) -> str:
    if path == "-":
        return sys.stdin.read()
    opener = gzip.open if path.endswith(".gz") else open
    mode = "rt" if path.endswith(".gz") else "r"
    with opener(path, mode) as f:
        return f.read()


def _read_records(path: str) -> list[dict]:
    """Parse ``path`` (or stdin) into raw, unvalidated ``{"sequence_id", "sequence"}`` records, in file
    order. FASTA (``>id ...`` headers) is detected by the first non-blank character; anything else is
    treated as plain text, one sequence per line. A record with no header token (or the plain-text
    fallback) gets ``sequence_id=None`` — left for ``AAPredictor.predict``'s own ``seq_<index>``
    auto-id rule (§9.1) to fill in, so ids are assigned identically whether a caller goes through this
    CLI or the ``predict_aa`` façade directly."""
    text = _read_text(path)
    stripped = text.lstrip()
    records: list[dict] = []
    if stripped.startswith(">"):
        name, buf = None, []
        for line in text.splitlines():
            if line.startswith(">"):
                if (name is not None or buf) and "".join(buf).strip():
                    records.append({"sequence_id": name, "sequence": "".join(buf)})
                tok = line[1:].split()
                name, buf = (tok[0] if tok else None), []
            else:
                buf.append(line.strip())
        if (name is not None or buf) and "".join(buf).strip():
            records.append({"sequence_id": name, "sequence": "".join(buf)})
    else:
        for line in text.splitlines():
            if line.strip():
                records.append({"sequence_id": None, "sequence": line.strip()})
    return records


def _write_run_metadata(out: str, predictor, args, counts: dict) -> None:
    prov = _provenance(predictor.card)
    meta = {
        **prov,
        "n_input": counts["n_input"],
        "n_predicted": counts["n_predicted"],
        "n_errors": counts["n_errors"],
        "command": {
            "input": args.input, "out": args.out, "model": args.model,
            "coverage_group": args.coverage_group, "on_invalid": args.on_invalid, "device": args.device,
        },
        "timestamp": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
    }
    with open(out + ".run.json", "w") as f:
        json.dump(meta, f, indent=2)


def run(args) -> int:
    records = _read_records(args.input)
    if not records:
        print(f"no sequences found in {args.input}")
        return 1

    predictor = load_aa_model(args.model, device=args.device)
    # Pass through the on_invalid flag from the CLI (default: "skip" for batch-friendly mode).
    results = predictor.predict(records, coverage_group=args.coverage_group, on_invalid=args.on_invalid)

    n_errors = 0
    with open(args.out, "w") as f:
        for r in results:
            if isinstance(r, ErrorResult):
                n_errors += 1
            f.write(json.dumps(r.to_dict()) + "\n")

    counts = {"n_input": len(records), "n_predicted": len(results) - n_errors, "n_errors": n_errors}
    if not args.no_run_metadata:
        _write_run_metadata(args.out, predictor, args, counts)

    print(f"predicted {counts['n_predicted']} sequences ({counts['n_errors']} errors) -> {args.out}")
    return 0
