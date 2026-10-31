"""lab-chip-v1 end to end on the human IGH reference: it trains from GenAIRR, reloads through a model file bound to its
reference, predicts AIRR rows and resumes. (The generic candidate contract in test_architectures assumes conv-v1's
components and both alphabets; the chip is nucleotide-only and is checked here.)"""
import numpy as np
import pytest

from alignair import TrainingConfig, run_training
from alignair.model_file import read_training_state

import alignair_lab.architectures  # noqa: F401  (registers lab-chip-v1)

pytestmark = pytest.mark.integration

OPTIONS = {"architecture": "lab-chip-v1", "embed_dim": 8, "boundary_mode": "half_open", "anchor_heads": True,
           "d_presence_head": True, "chip_inputs": True}


def test_a_tiny_chip_trains_reloads_predicts_airr_rows_and_resumes(tmp_path):
    request = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", steps=2, batch_size=2, device="cpu", seed=31,
                                          val_every=1, model_options=OPTIONS)
    losses, metrics = [], []
    run = run_training(request, output_dir=tmp_path / "first",
                       on_iteration=lambda item: losses.append(item.result.total))
    assert len(losses) == 2 and np.isfinite(losses).all()
    aligner = run.best_aligner(device="cpu")
    assert aligner.model.cfg.architecture == "lab-chip-v1" and aligner.model.core.bank is not None
    sequences = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTC",
                 "ACGTN"]
    result = aligner.predict(sequences, batch_size=2)
    rows = list(result)
    assert len(rows) == 2 and all(row["v_call"] for row in rows)
    raw = aligner.predict_raw(sequences, batch_size=1)
    assert raw["v_allele"].shape == (2, aligner.model.cfg.v_allele_count)
    assert not any(key.startswith("chip_") for key in raw)
    steps = []
    resumed = run_training(request.with_(steps=3, resume_path=run.model_path), output_dir=tmp_path / "resumed",
                           on_iteration=lambda item: steps.append(item.step))
    assert steps == [3] and read_training_state(resumed.model_path).step == 3
