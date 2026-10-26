"""The alignair-api runner's subprocess: AlignAIR's public Python API run on a FASTA, written as ``alignair predict``
writes it.

    python -m alignair_assay.diagnostics.api_predict --checkpoint MODEL.alignair --reference REFERENCE.fasta \\
        --input READS.fasta --out OUT.tsv [--device cpu] [--batch-size 64] [--genotype GENOTYPE.json]

``runners.alignair_api`` runs it in a subprocess, by module name: a runner may not import ``alignair``, and the assay
process never imports torch. It does what ``alignair/cli/predict.py`` does, through the public API, so the shared
and the genotype arm run the same code and model:

1. The checkpoint is loaded with ``diagnostics.replay.load_model_checked``: ``alignair.api.load_model``, as the CLI's
   ``Aligner.from_pretrained`` loads it, on the device ``alignair.resolve_device`` makes of ``--device``, as the
   CLI's does. The model's reference must be the one in ``--reference``, which the runner verified against the
   truth reference.
2. ``--genotype``, a JSON ``{"v": [...], "d": [...], "j": [...]}``, is read with AlignAIR's
   ``genotype.constraint.load_genotype(path, reference=<the model's>, drop_unknown=False)``. It returns
   ``(genotype, unknown)`` and raises on an allele the model does not hold. Its ``genotype``, ``{gene letter: set of
   allele names}``, is the type ``predict_sequences`` takes (``genotype=`` becomes ``PredictConfig.genotype``), and it
   goes with ``genotype_method="mask"``, the CLI's default. The CLI's ``predict --genotype`` passes the whole pair on
   as the genotype, and crashes (``cli/predict.py`` in 3.0.0).
3. The reads are read with ``alignair.iter_sequences`` in chunks of ``CHUNK_SIZE``, the CLI's default
   ``--chunk-size``, so the cleaned reads, their ids and the batches are the CLI's.
4. Each chunk is predicted with ``alignair.predict_sequences(model, reference, sequences, device=, batch_size=,
   airr=True, genotype=, genotype_method=)``, the genotype None in the shared arm.
5. The rows are written with ``alignair.AirrWriter``, as the CLI writes them: atomically, every column, under the
   model's locus when it declares one (else IGH).

``<out>.run.json`` records the run, as the CLI's sidecar does: ``api``, ``checkpoint``, ``reference_digest``,
``device`` (resolved), ``torch``, ``cuda_device`` (None on a CPU device), ``batch_size``, ``chunk_size``, ``locus``,
``genotype`` (its ``path``, ``method`` and allele counts per gene letter; None without one) and the CLI sidecar's
``counts``: input, accepted, rejected and written records, and complete, partial and failed AIRR assemblies. Unlike
the CLI, no rate of failed assemblies fails the run; the counts are recorded instead.

Exit status: 0 once the TSV and the record are written; 1 when no read is valid (an empty TSV is written, as the CLI
writes it); 3 (``GENOTYPE_FAILED``) when a genotype was given and reading it (``load_genotype``) or computing its mask
on the model's reference (``genotype_allowed_mask``, which the prediction computes for each chunk; here once, before
any read) raised one of ``GENOTYPE_ERRORS`` (AlignAIR's ``NovelAlleleUnsupportedError``, or the ``AttributeError``,
``TypeError`` or ``KeyError`` a genotype of another type raises where it is applied), after the traceback and a last
line ``genotype API path failed: <type>: <message>`` on stderr. Any other error propagates, with or without a
genotype (Python's traceback, exit status 1, no note), and so does any error of the prediction itself, of those types
too. A run that raises writes no TSV and no record.
"""
from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import torch
from alignair import AirrWriter, iter_sequences, predict_sequences, resolve_device
from alignair.genotype.constraint import NovelAlleleUnsupportedError, genotype_allowed_mask, load_genotype

from alignair_assay.core.reference import Reference
from alignair_assay.diagnostics.replay import load_model_checked

# The CLI's default --chunk-size: reads are predicted this many at a time.
CHUNK_SIZE = 20000
# The CLI's default --genotype-method: every allele outside the genotype gets probability 0.
GENOTYPE_METHOD = "mask"
# The exit status, and the last line's prefix, of a failure on the genotype path.
GENOTYPE_FAILED = 3
GENOTYPE_NOTE = "genotype API path failed"
# The errors that are the genotype path's: AlignAIR's refusal of a genotype the model cannot take, and what a
# genotype of another type than ``predict_sequences`` expects raises where it is applied. Any other error is the
# tool's, with or without a genotype.
GENOTYPE_ERRORS = (NovelAlleleUnsupportedError, AttributeError, TypeError, KeyError)
API = "alignair.predict_sequences"
COUNTS = ("input_records", "accepted_records", "rejected_records", "written_records", "complete_assemblies",
          "partial_assemblies", "failed_assemblies")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m alignair_assay.diagnostics.api_predict",
                                description="AlignAIR's public API on a FASTA, written as an AIRR TSV")
    p.add_argument("--checkpoint", required=True, help="AlignAIR .alignair model file")
    p.add_argument("--reference", required=True, help="FASTA of the reference the model must hold")
    p.add_argument("--input", required=True, help="reads (FASTA)")
    p.add_argument("--out", required=True, help="AIRR TSV to write")
    p.add_argument("--device", default="cpu", help="torch device (cpu, cuda, auto)")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--genotype", help='genotype JSON {"v": [...], "d": [...], "j": [...]} to restrict the calls to')
    return p


def main(argv=None) -> int:
    """Run the module docstring's steps on the parsed arguments ``argv`` (``sys.argv[1:]`` when None); returns the
    exit status."""
    args = build_parser().parse_args(argv)
    device = resolve_device(args.device)
    truth = Reference.from_fasta(args.reference)
    model, reference = load_model_checked(args.checkpoint, truth, device=device)
    loci = reference.locus_names() if hasattr(reference, "locus_names") else ()
    locus = loci[0] if len(loci) == 1 else "IGH"
    out = Path(args.out)
    sidecar = out.with_name(out.name + ".run.json")
    sidecar.unlink(missing_ok=True)
    record = {"api": API, "checkpoint": str(Path(args.checkpoint).absolute()), "reference_digest": truth.digest(),
              "device": device, "torch": torch.__version__, "cuda_device": _cuda_device(device),
              "batch_size": args.batch_size, "chunk_size": CHUNK_SIZE, "locus": locus, "genotype": None}
    if args.genotype is None:
        counts = predict_file(model, reference, args.input, out, device=device, batch_size=args.batch_size,
                              locus=locus)
    else:
        try:  # the genotype path's errors: reading the genotype and computing its mask, before any read
            genotype, _ = load_genotype(args.genotype, reference=reference, drop_unknown=False)
            genotype_allowed_mask(genotype, reference)
        except GENOTYPE_ERRORS as error:
            traceback.print_exc()
            print(f"{GENOTYPE_NOTE}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            return GENOTYPE_FAILED
        counts = predict_file(model, reference, args.input, out, device=device, batch_size=args.batch_size,
                              locus=locus, genotype=genotype)
        record["genotype"] = {"path": args.genotype, "method": GENOTYPE_METHOD,
                              "alleles": {letter: len(names) for letter, names in genotype.items()}}
    record["counts"] = counts
    sidecar.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    if counts["accepted_records"] == 0:
        print(f"no valid reads in {args.input} ({counts['rejected_records']} dropped)", file=sys.stderr)
        return 1
    print(f"aligned {counts['accepted_records']} reads ({counts['rejected_records']} dropped) -> {out}")
    return 0


def predict_file(model, reference, fasta, out, *, device: str, batch_size: int, locus: str,
                 genotype: dict | None = None) -> dict[str, int]:
    """Predict every read of ``fasta`` with ``predict_sequences`` and write the rows to ``out`` (steps 3 to 5 of the
    module docstring); returns the ``COUNTS``. ``out`` is replaced only when every chunk was written."""
    counts = dict.fromkeys(COUNTS, 0)
    with AirrWriter(str(out), locus=locus) as writer:
        for ids, sequences, dropped in iter_sequences(str(fasta), chunk_size=CHUNK_SIZE):
            counts["rejected_records"] += dropped
            if not sequences:
                continue
            records = predict_sequences(model, reference, sequences, device=device, batch_size=batch_size,
                                        airr=True, genotype=genotype, genotype_method=GENOTYPE_METHOD)
            writer.write(ids, sequences, records)
            counts["accepted_records"] += len(sequences)
            counts["written_records"] += len(records)
            for row in records:
                status = row.get("airr_assembly_status")
                if status in ("complete", "partial", "failed"):
                    counts[f"{status}_assemblies"] += 1
    counts["input_records"] = counts["accepted_records"] + counts["rejected_records"]
    return counts


def _cuda_device(device: str) -> str | None:
    """The name of CUDA device ``device``; None for another device."""
    found = torch.device(device)
    return torch.cuda.get_device_name(found) if found.type == "cuda" else None


if __name__ == "__main__":
    sys.exit(main())
