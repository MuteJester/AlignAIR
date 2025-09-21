import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import tensorflow as tf


class SavedModelInferenceWrapper:
    """
    Lightweight adapter around a TensorFlow SavedModel for inference.

    Exposes a .predict(inputs) method compatible with the pipeline, plus a few
    metadata attributes read from the bundle's config.json for downstream steps.
    """

    def __init__(self, saved_model_dir: str | Path,
                 bundle_dir: str | Path,
                 config: Optional[Dict[str, Any]] = None) -> None:
        self.saved_model_dir = Path(saved_model_dir)
        self.bundle_dir = Path(bundle_dir)

        # Load SavedModel once
        self._sm = tf.saved_model.load(str(self.saved_model_dir))
        # Prefer named signature; fall back to the only signature if needed
        sig = getattr(self._sm, 'signatures', None)
        if sig and 'serving_default' in sig:
            self._fn = sig['serving_default']
        else:
            # Take any available signature
            keys = list(sig.keys()) if sig else []
            if not keys:
                raise RuntimeError("SavedModel has no callable signatures")
            self._fn = sig[keys[0]]

        # Load config for metadata
        cfg = config
        if cfg is None:
            cfg_path = self.bundle_dir / 'config.json'
            if cfg_path.exists():
                with open(cfg_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
            else:
                cfg = {}

        # Expose commonly used attributes
        self.max_seq_length: int = int(cfg.get('max_seq_length', 576))
        self.has_d_gene: bool = bool(cfg.get('has_d_gene', False))
        self.v_allele_count: Optional[int] = cfg.get('v_allele_count')
        self.j_allele_count: Optional[int] = cfg.get('j_allele_count')
        self.d_allele_count: Optional[int] = cfg.get('d_allele_count')
        # Attach dataconfig(s) lazily by loader as attributes when available
        # For single-chain models, loaders will set `dataconfig` (a DataConfig)
        # For multi-chain models, loaders will set `dataconfigs` (a MultiDataConfigContainer)
        self.dataconfig = None
        self.dataconfigs = None

    def predict(self, inputs: Dict[str, np.ndarray], verbose: int = 0, batch_size: Optional[int] = None) -> Dict[str, np.ndarray]:
        """
        Run inference using the SavedModel signature. Accepts a dict with
        'tokenized_sequence' as a numpy array. Returns a dict of numpy arrays.
        """
        if 'tokenized_sequence' not in inputs:
            raise KeyError("predict expects inputs with key 'tokenized_sequence'")

        x = inputs['tokenized_sequence']
        # Ensure tensor and dtype are compatible with the exported graph
        # Export used int32 when building; cast defensively.
        x_tensor = tf.convert_to_tensor(x)
        if x_tensor.dtype not in (tf.int32, tf.int64):
            x_tensor = tf.cast(x_tensor, tf.int32)

        # Call the signature with a named argument
        outputs = self._fn(tokenized_sequence=x_tensor)

        # Convert to plain numpy dict
        result: Dict[str, np.ndarray] = {}
        for k, v in outputs.items():
            try:
                result[k] = v.numpy()
            except Exception:
                # In rare cases it may already be numpy
                result[k] = np.array(v)
        return result
