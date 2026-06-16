"""Numeric-equivalence checks: new PyTorch math vs legacy TensorFlow math.

These import TensorFlow and reproduce the legacy math (identical to the formulas
in SingleChainAlignAIR). They are skipped automatically if TF import fails.
"""
import numpy as np
import pytest
import torch

tf = pytest.importorskip("tensorflow")


def test_soft_targets_match_tf():
    from alignair.losses.functional import soft_targets as pt_soft

    L, gt, sigma = 12, 5.0, 1.5
    pt = pt_soft(torch.tensor([[gt]]), L=L, sigma=sigma).numpy()[0]

    # Reproduce the legacy TF soft_targets math inline (same as SingleChainAlignAIR).
    gt_t = tf.constant([[gt]], dtype=tf.float32)
    positions = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    dist2 = tf.square(positions - gt_t)
    logits = -0.5 * dist2 / (sigma * sigma)
    tf_probs = tf.nn.softmax(logits, axis=-1).numpy()[0]

    np.testing.assert_allclose(pt, tf_probs, atol=1e-5)


def test_soft_label_ce_matches_tf():
    from alignair.losses.functional import soft_label_cross_entropy, soft_targets

    L = 12
    rng = np.random.default_rng(0)
    logits_np = rng.standard_normal((4, L)).astype(np.float32)
    gt = np.array([[2.0], [5.0], [9.0], [0.0]], dtype=np.float32)

    t_pt = soft_targets(torch.tensor(gt), L=L)
    pt_loss = soft_label_cross_entropy(t_pt, torch.tensor(logits_np)).item()

    positions = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    gt_t = tf.constant(gt)
    dist2 = tf.square(positions - gt_t)
    t_tf = tf.nn.softmax(-0.5 * dist2 / (1.5 * 1.5), axis=-1)
    tf_loss = float(tf.reduce_mean(
        tf.nn.softmax_cross_entropy_with_logits(labels=t_tf, logits=tf.constant(logits_np))))

    assert abs(pt_loss - tf_loss) < 1e-4


def test_expectation_matches_tf():
    from alignair.losses.functional import expectation_from_logits

    L = 12
    rng = np.random.default_rng(1)
    logits_np = rng.standard_normal((3, L)).astype(np.float32)

    pt = expectation_from_logits(torch.tensor(logits_np), max_seq_length=L).numpy().ravel()

    probs = tf.nn.softmax(tf.constant(logits_np), axis=-1)
    pos = tf.cast(tf.range(L), tf.float32)[tf.newaxis, :]
    tf_exp = tf.reduce_sum(probs * pos, axis=-1).numpy()

    np.testing.assert_allclose(pt, tf_exp, atol=1e-4)


def test_entropy_metric_matches_tf():
    from alignair.metrics.entropy import AlleleEntropy

    rng = np.random.default_rng(2)
    probs_np = rng.random((5, 7)).astype(np.float32)
    probs_np /= probs_np.sum(axis=1, keepdims=True)

    ent = AlleleEntropy()
    ent.update(torch.tensor(probs_np))
    pt_val = ent.compute().item()

    tf_entropy = -tf.reduce_sum(
        tf.constant(probs_np) * tf.math.log(tf.constant(probs_np) + 1e-9), axis=-1)
    tf_val = float(tf.reduce_mean(tf_entropy))

    assert abs(pt_val - tf_val) < 1e-4
