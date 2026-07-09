"""Data augmentation and target-building transformation for training records."""

import numpy as np

from .crop import crop_record
from .targets import build_targets, _tok

# token complement by id: pad0 A1<->T2 G3<->C4 N5
_COMPLEMENT_NP = np.array([0, 2, 1, 4, 3, 5], dtype=np.int64)


def _orient_tokens(tokens, t):
    """Apply orientation transform t in {0:id,1:revcomp,2:comp,3:reverse} to a 1-D
    numpy token array. All transforms are involutions (re-applying recovers forward)."""
    if t == 1:
        return _COMPLEMENT_NP[tokens][::-1].copy()
    if t == 2:
        return _COMPLEMENT_NP[tokens].copy()
    if t == 3:
        return tokens[::-1].copy()
    return tokens


class GymRecordTransformer:
    """Encapsulates data augmentation (cropping and orientation) and target building for a single simulated record."""

    def __init__(self, reference_set):
        self.reference_set = reference_set

    def transform(self, record: dict, params: dict, has_d: bool, rng: np.random.Generator) -> dict:
        # teacher (EMA self-distillation) view: the full, forward read of THIS record,
        # before the student's crop/orientation augmentation.
        teacher_tokens = _tok(str(record["sequence"]).upper())
        if params.get("crop_prob", 0) > 0 and rng.random() < params["crop_prob"]:
            lo, hi = params["crop_len_min"], params["crop_len_max"]
            if params.get("crop_log_uniform"):   # densely sample SHORT fragments
                target_len = int(round(np.exp(rng.uniform(np.log(lo), np.log(hi)))))
            else:
                target_len = int(rng.integers(lo, hi + 1))
            record = crop_record(record, target_len)
        bundle = build_targets(record, self.reference_set, has_d=has_d)
        # present a fraction of reads in a non-forward orientation; targets stay in
        # forward frame (the model canonicalizes), only orientation_id changes.
        if params.get("orient_prob", 0) > 0 and rng.random() < params["orient_prob"]:
            t = int(rng.integers(1, 4))  # 1=revcomp, 2=comp, 3=reverse
            bundle["tokens"] = _orient_tokens(bundle["tokens"], t)
            bundle["orientation_id"] = t
        bundle["teacher_tokens"] = teacher_tokens
        return bundle
