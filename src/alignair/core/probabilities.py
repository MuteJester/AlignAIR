"""NumPy categorical probability/target contracts shared by calibration and evaluation."""
import numpy as np

from ..errors import ConfigurationError, InputError, NumericalError


def validate_categorical_probabilities(values, dimensions=2):
    """Validate a nonempty categorical vector/matrix without copying or renormalizing it."""
    if type(dimensions) is not int or dimensions not in (1, 2):
        raise ConfigurationError("categorical probabilities require one or two dimensions", context={"field": "dimensions"})
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise InputError("probabilities must form a numeric array", context={"field": "probabilities"}) from exc
    if values.ndim != dimensions or not all(values.shape) or values.dtype.kind not in "iuf":
        raise InputError("probabilities have an invalid shape or numeric type", context={"field": "probabilities"})
    if not np.isfinite(values).all():
        raise NumericalError("non-finite class probabilities", context={"field": "probabilities"})
    epsilon = np.finfo(values.dtype).eps if values.dtype.kind == "f" else 0.
    if (np.any(values < 0) or np.any(values > 1)
            or not np.allclose(values.sum(axis=-1, dtype=np.float64), 1., rtol=0., atol=max(1e-6, 4 * epsilon))):
        raise InputError("class probabilities must lie in [0, 1] and sum to one", context={"field": "probabilities"})
    return values


def validate_categorical_targets(values, probability_shape):
    """Validate one observed integer class ID per row of an already-validated matrix."""
    try:
        values = np.asarray(values)
    except (TypeError, ValueError) as exc:
        raise InputError("categorical truth must form an integer vector", context={"field": "true"}) from exc
    if (values.shape != (probability_shape[0],) or values.dtype.kind not in "iu"
            or np.any(values < 0) or np.any(values >= probability_shape[1])):
        raise InputError("categorical truth must contain one valid integer class ID per row", context={"field": "true"})
    return values
