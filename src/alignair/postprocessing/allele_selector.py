"""Allele selection by max-likelihood percentage threshold (port of
MaxLikelihoodPercentageThreshold), decoupled from DataConfig."""
import numpy as np


def max_likelihood_threshold(prediction: np.ndarray, percentage: float = 0.21, cap: int = 3):
    """Select indices with likelihood >= percentage*max, sorted desc, capped at `cap`."""
    max_index = int(np.argmax(prediction))
    threshold_value = prediction[max_index] * percentage
    indices = np.where(prediction >= threshold_value)[0]
    indices = indices[np.argsort(-prediction[indices])]
    if len(indices) > cap:
        indices = indices[:cap]
    return indices, prediction[indices]


def select_alleles(prob_matrix: np.ndarray, index_to_allele: dict,
                   percentage: float = 0.21, cap: int = 3):
    """For each row, return (allele_names, likelihoods)."""
    results = []
    for vec in prob_matrix:
        idx, lik = max_likelihood_threshold(vec, percentage=percentage, cap=cap)
        results.append(([index_to_allele[int(i)] for i in idx], lik))
    return results
