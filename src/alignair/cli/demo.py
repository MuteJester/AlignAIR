"""``alignair demo`` — the documented first-run path, end to end, with no published model and no
network: train a tiny model, predict a few simulated reads, and also predict them under a donor
(genotype-subset) reference so the two runs can be compared.

Produces, under ``--out``:  ``bundle/`` (the trained model), ``demo.tsv`` (default-reference
predictions), and ``demo_donor.tsv`` (donor-constrained predictions). Kept small on purpose so it runs
on CPU in seconds (``--steps 1`` in the release smoke test); it exercises train → save → load →
predict → genotype-constrain in one command.
"""
from __future__ import annotations

import itertools
import os


def register(sub) -> None:
    p = sub.add_parser("demo", help="tiny end-to-end train -> predict -> donor-compare (no network)")
    p.add_argument("-o", "--out", default=None,
                   help="output directory for the demo artifacts (default: a fresh temp directory)")
    p.add_argument("--dataconfig", default="HUMAN_IGH_OGRDB", help="GenAIRR dataconfig to demo on")
    p.add_argument("--steps", type=int, default=5, help="training steps (tiny; 1 in the smoke test)")
    p.add_argument("--reads", type=int, default=8, help="number of simulated reads to predict")
    p.add_argument("--device", default=None, help="cpu / cuda (default: auto)")
    p.set_defaults(func=run)


def _simulate_reads(dataconfig, n: int, seed: int = 7) -> list:
    """A handful of simulated reads for the demo (moderate SHM), via the training gym."""
    from ..train.gym import Curriculum, build_experiment
    exp = build_experiment(dataconfig, dict(Curriculum().params(0.3)), allow_curatable=True)
    return [str(r["sequence"]).upper()
            for r in itertools.islice(exp.stream_records(n=None, seed=seed), n)]


def run(args) -> int:
    import torch
    import GenAIRR.data as gd
    from ..api import load_model, predict_sequences, train_model
    from ..io.airr import write_airr

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = args.out or __import__("tempfile").mkdtemp(prefix="alignair_demo_")  # zero-arg `alignair demo`
    os.makedirs(out, exist_ok=True)
    bundle = os.path.join(out, "bundle")
    os.makedirs(bundle, exist_ok=True)
    model_path = os.path.join(bundle, "model.alignair")
    dc = getattr(gd, args.dataconfig)

    print(f"[demo] training a tiny model on {args.dataconfig} ({args.steps} steps, {device})...")
    train_model([args.dataconfig], out_path=model_path, steps=args.steps, device=device)

    print("[demo] simulating reads + predicting (default reference)...")
    reads = _simulate_reads(dc, args.reads)
    ids = [f"demo{i}" for i in range(len(reads))]
    model, ref = load_model(model_path, device=device)
    records = predict_sequences(model, ref, reads, device=device)
    write_airr(os.path.join(out, "demo.tsv"), ids, reads, records, locus="IGH")

    print("[demo] predicting under a donor (genotype-subset) reference...")
    donor = {g: set(list(ref.gene(g.upper()).names)[: max(2, len(ref.gene(g.upper())) // 3)])
             for g in (("v", "d", "j") if ref.has_d else ("v", "j"))}
    donor_records = predict_sequences(model, ref, reads, device=device,
                                      genotype=donor, genotype_method="mask")
    write_airr(os.path.join(out, "demo_donor.tsv"), ids, reads, donor_records, locus="IGH")

    print(f"[demo] done -> {out}  (bundle/, demo.tsv, demo_donor.tsv)")
    return 0
