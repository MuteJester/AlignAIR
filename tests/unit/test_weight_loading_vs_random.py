import os
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from AlignAIR.Serialization.io import load_bundle
from AlignAIR.Models.SingleChainAlignAIR.SingleChainAlignAIR import SingleChainAlignAIR

pytestmark = [pytest.mark.unit]


def _sum_l2(weights):
    total = 0.0
    for w in weights:
        try:
            total += float(tf.norm(w).numpy())
        except Exception:
            pass
    return total


def _sum_l2_diff(a, b):
    total = 0.0
    for wa, wb in zip(a, b):
        if wa.shape != wb.shape:
            raise AssertionError(f"Mismatched weight shapes: {wa.shape} vs {wb.shape}")
        try:
            total += float(tf.norm(wa - wb).numpy())
        except Exception:
            pass
    return total


@pytest.mark.skipif(
    not (Path(__file__).resolve().parents[2] / 'checkpoints' / 'AlignAIR_IGH_Extended' / 'config.json').exists(),
    reason='AlignAIR_IGH_Extended bundle not available'
)
def test_loaded_bundle_weights_are_not_random():
    repo_root = Path(__file__).resolve().parents[2]
    bundle_dir = repo_root / 'checkpoints' / 'AlignAIR_IGH_Extended'

    # Load bundle config and dataconfig
    cfg, dataconfig_obj, _meta = load_bundle(bundle_dir)

    # Build a randomly initialized model with the same architecture
    rand_model = SingleChainAlignAIR(
        max_seq_length=cfg.max_seq_length,
        dataconfig=dataconfig_obj,
        v_allele_latent_size=cfg.v_allele_latent_size,
        d_allele_latent_size=cfg.d_allele_latent_size,
        j_allele_latent_size=cfg.j_allele_latent_size,
    )
    _ = rand_model({"tokenized_sequence": tf.zeros((1, cfg.max_seq_length), dtype=tf.float32)}, training=False)

    # Load the pretrained model from the bundle
    loaded_model = SingleChainAlignAIR.from_pretrained(bundle_dir)

    # Ensure same number of variables and pairwise comparable shapes
    rand_weights = [w for w in rand_model.trainable_weights]
    loaded_weights = [w for w in loaded_model.trainable_weights]
    assert len(rand_weights) == len(loaded_weights), (
        f"Different number of trainable weights: random={len(rand_weights)}, loaded={len(loaded_weights)}"
    )
    for i, (ra, lo) in enumerate(zip(rand_weights, loaded_weights)):
        assert ra.shape == lo.shape, f"Mismatch at index {i}: {ra.shape} vs {lo.shape}"

    # Compute norms
    loaded_norm = _sum_l2(loaded_weights)
    rand_norm = _sum_l2(rand_weights)
    diff_norm = _sum_l2_diff(loaded_weights, rand_weights)

    # Sanity: norms should be non-trivial
    assert loaded_norm > 10.0, f"Loaded model weight norm too small ({loaded_norm:.6f})."
    assert rand_norm > 10.0, f"Random model weight norm too small ({rand_norm:.6f})."

    # The loaded model should be significantly different from a random init
    rel_diff_loaded = diff_norm / (loaded_norm + 1e-8)
    rel_diff_rand = diff_norm / (rand_norm + 1e-8)

    # Expect large relative differences because random init shouldn't match trained weights
    assert rel_diff_loaded > 0.1, (
        f"Loaded-vs-random relative diff too small: {rel_diff_loaded:.6f}; this suggests weights might not be loaded."
    )
    assert rel_diff_rand > 0.1, (
        f"Random-vs-loaded relative diff too small: {rel_diff_rand:.6f}; this suggests weights might not be loaded."
    )
