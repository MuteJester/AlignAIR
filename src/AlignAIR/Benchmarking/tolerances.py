"""
Default tolerance configurations for snapshot comparisons.

Three preset profiles:
- CODE_CHANGE_TOLERANCES: Same model weights, only code changed. Very strict.
- DEFAULT_TOLERANCES: General-purpose defaults.
- MODEL_COMPARISON_TOLERANCES: Comparing two different trained models. Looser.
"""

CODE_CHANGE_TOLERANCES = {
    "predictions": {
        "allele_classification_atol": 1e-5,
        "position_atol": 1e-5,
        "scalar_atol": 1e-5,
    },
    "latent": {
        "cosine_similarity_min": 0.99999,
        "mean_vector_l2_max": 1e-5,
    },
    "metrics": {
        "accuracy_atol": 0.001,
        "boundary_mae_atol": 0.01,
    },
    "pipeline": {
        "allele_call_match_rate_min": 1.0,
        "position_exact_match_min": 1.0,
        "numeric_atol": 1e-5,
    },
    "training": {
        "accuracy_atol": 0.001,
        "boundary_mae_atol": 0.01,
        "loss_rtol": 0.001,
    },
}

DEFAULT_TOLERANCES = {
    "predictions": {
        "allele_classification_atol": 1e-4,
        "position_atol": 1e-4,
        "scalar_atol": 1e-3,
    },
    "latent": {
        "cosine_similarity_min": 0.999,
        "mean_vector_l2_max": 1e-3,
    },
    "metrics": {
        "accuracy_atol": 0.02,
        "boundary_mae_atol": 1.0,
    },
    "pipeline": {
        "allele_call_match_rate_min": 0.98,
        "position_exact_match_min": 0.95,
        "numeric_atol": 1e-3,
    },
    "training": {
        "accuracy_atol": 0.02,
        "boundary_mae_atol": 1.0,
        "loss_rtol": 0.10,
    },
}

MODEL_COMPARISON_TOLERANCES = {
    "predictions": {
        "allele_classification_atol": 0.1,
        "position_atol": 5.0,
        "scalar_atol": 0.05,
    },
    "latent": {
        "cosine_similarity_min": 0.80,
        "mean_vector_l2_max": 1.0,
    },
    "metrics": {
        "accuracy_atol": 0.05,
        "boundary_mae_atol": 3.0,
    },
    "pipeline": {
        "allele_call_match_rate_min": 0.90,
        "position_exact_match_min": 0.85,
        "numeric_atol": 0.05,
    },
    "training": {
        "accuracy_atol": 0.05,
        "boundary_mae_atol": 3.0,
        "loss_rtol": 0.20,
    },
}


def get_tolerances(profile: str = "default") -> dict:
    """Return a tolerance dict by profile name."""
    profiles = {
        "code-change": CODE_CHANGE_TOLERANCES,
        "default": DEFAULT_TOLERANCES,
        "model-comparison": MODEL_COMPARISON_TOLERANCES,
    }
    if profile not in profiles:
        raise ValueError(f"Unknown tolerance profile '{profile}'. Choose from: {list(profiles.keys())}")
    return profiles[profile]
