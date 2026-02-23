"""Determinism lock — ensure reproducible results across runs."""
from __future__ import annotations

import hashlib
import os
import random
from typing import Dict


def set_deterministic(seed: int = 42, tf_deterministic: bool = True) -> Dict[str, str]:
    """Lock all sources of non-determinism.

    Sets: PYTHONHASHSEED, random.seed, np.random.seed,
    tf.random.set_seed, and optionally TF_DETERMINISTIC_OPS.

    Returns:
        Dict of settings applied (for provenance).
    """
    settings: Dict[str, str] = {}

    # Python hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    settings["PYTHONHASHSEED"] = str(seed)

    # stdlib random
    random.seed(seed)
    settings["random_seed"] = str(seed)

    # numpy
    import numpy as np
    np.random.seed(seed)
    settings["numpy_seed"] = str(seed)

    # tensorflow
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        settings["tf_seed"] = str(seed)

        if tf_deterministic:
            os.environ["TF_DETERMINISTIC_OPS"] = "1"
            settings["TF_DETERMINISTIC_OPS"] = "1"
            try:
                tf.config.experimental.enable_op_determinism()
                settings["tf_op_determinism"] = "enabled"
            except AttributeError:
                pass
    except ImportError:
        pass

    return settings
