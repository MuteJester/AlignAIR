"""The byte pin of what the default recipe trains on: its first 64 records, the two batches the training adapter makes
of them, and the loss and gradient norm of one training step on the first (``PIN``,
``tests/alignair/fixtures/training_pin_v3.json``). ``python -m tests.support.training_pin`` measures them on the tree
it runs on; a moved digest is a moved training batch."""
from __future__ import annotations

import hashlib
import itertools
import json
import subprocess
import sys
from pathlib import Path

PIN_PATH = Path(__file__).resolve().parents[1] / "alignair" / "fixtures" / "training_pin_v3.json"
PIN = json.loads(PIN_PATH.read_text()) if PIN_PATH.exists() and PIN_PATH.stat().st_size else None
SMALL = {"embed_dim": 4, "trunk_channels": 8}


def _policy():
    from alignair.data.flanks import FlankPolicy
    from alignair.train.gym.streams import CurriculumStreamPolicy
    return CurriculumStreamPolicy(progresses=(0.3, 0.6, 0.9), heavy_shm=0.25, short_boost=1, flanks=FlankPolicy())


def _setup():
    import GenAIRR.data as gd
    from alignair.core import prepare_model
    from alignair.reference.reference_set import ReferenceSet
    return prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), input_space="nucleotide",
                         label_space="alleles", **SMALL)


def pinned_records() -> list[dict]:
    """The first 64 records of the lab mix (with ``TrainingConfig``'s default flanks) on HUMAN_IGH_OGRDB, seed 0."""
    import GenAIRR.data as gd
    from alignair.train.gym.streams import training_records
    return [dict(r) for r in itertools.islice(training_records((gd.HUMAN_IGH_OGRDB,), policy=_policy(), seed=0), 64)]


def records_digest(records) -> str:
    return hashlib.sha256(json.dumps(records, sort_keys=True).encode("utf-8")).hexdigest()


def pinned_batches(records) -> list:
    """The training adapter's batches of ``records[:32]`` and ``records[32:]``, torch seeded 1234 first."""
    import torch
    from alignair.data.training import prepare_training_data
    torch.manual_seed(1234)
    adapter = prepare_training_data(_setup()).batch_adapter
    return [adapter(records[:32]), adapter(records[32:])]


def batch_digest(batch) -> str:
    """One SHA-256 over the batch's inputs, then its targets, keys sorted: each tensor's part/key/dtype/shape, then
    its bytes."""
    digest = hashlib.sha256()
    for part, tensors in (("inputs", batch.inputs), ("targets", batch.targets)):
        for key in sorted(tensors):
            tensor = tensors[key].detach().cpu().contiguous()
            digest.update(f"{part}/{key}/{tensor.dtype}/{tuple(tensor.shape)}".encode())
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def first_step(batch):
    """One training step of a fresh small chip (torch seeded 0) on ``batch`` (seeded 1234), on one CPU thread."""
    import torch
    from alignair.tasks import build_tasks
    from alignair.train.setup import build_training_engine
    torch.set_num_threads(1)
    setup = _setup()
    torch.manual_seed(0)
    model = setup.build_model()
    engine = build_training_engine(model, tasks=build_tasks(setup.config), device="cpu", lr=3e-4)
    torch.manual_seed(1234)
    return engine.step(batch, step=1)


def measure() -> dict:
    records = pinned_records()
    batches = pinned_batches(records)
    step = first_step(batches[0])
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    return {"schema": "alignair.training-pin.v1", "commit": head, "genairr_freeze": "c4c73b62",
            "records_sha256": records_digest(records), "batch_sha256": [batch_digest(b) for b in batches],
            "step": {"total": step.total, "gradient_norm": step.gradient_norm, "parts": dict(sorted(step.parts.items()))}}


if __name__ == "__main__":
    json.dump(measure(), sys.stdout, indent=1, sort_keys=True)
    sys.stdout.write("\n")
