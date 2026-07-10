"""User-facing façade for AlignAIR — load a trained model and run predictions (or train a custom
model) in a few lines, without touching the internal config/pipeline machinery.

    from alignair import load_model, predict_sequences
    model, reference = load_model("model.pt", dataconfigs=["HUMAN_IGH_OGRDB"])
    records = predict_sequences(model, reference, ["CAGGTGCAGCTG..."])
    # records: AIRR-convention dicts (v_call/d_call/j_call, coordinates, CIGAR, productivity, ...)

    from alignair import train_model
    train_model(["HUMAN_IGH_OGRDB"], out_path="my_model.pt", steps=100_000)
"""
from __future__ import annotations

import re
from typing import Sequence

import torch

from .core.config import AlignAIRConfig
from .core import AlignAIR
from .predict import PredictConfig, predict as _predict
from .reference.reference_set import ReferenceSet

__all__ = ["load_model", "predict_sequences", "train_model"]

# pre-unify SingleChain/MultiChain checkpoint keys -> unified AlignAIR keys (no-op for current models)
_LEGACY_KEY_REWRITES = [
    (r"seg_towers\.([vdj])\.(.*)", r"branches.\1.seg_tower.\2"),
    (r"seg_heads\.([vdj])_(start|end)\.(.*)", r"branches.\1.\2_head.\3"),
    (r"cls_towers\.([vdj])\.(.*)", r"branches.\1.cls_tower.\2"),
    (r"cls_mid\.([vdj])\.(.*)", r"branches.\1.cls_mid.\2"),
    (r"cls_head\.([vdj])\.(.*)", r"branches.\1.cls_head.\2"),
    (r"mutation_rate_(mid|head)\.(.*)", r"meta_heads.mutation_rate.\1.\2"),
    (r"indel_count_(mid|head)\.(.*)", r"meta_heads.indel_count.\1.\2"),
    (r"productive_head\.(.*)", r"meta_heads.productive.head.\1"),
    (r"chain_type_(mid|head)\.(.*)", r"meta_heads.chain_type_logits.\1.\2"),
]


def _remap_state_dict(state: dict) -> dict:
    out = {}
    for key, value in state.items():
        new_key = key
        for pattern, repl in _LEGACY_KEY_REWRITES:
            if re.match(pattern, key):
                new_key = re.sub(pattern, repl, key)
                break
        out[new_key] = value
    return out


def _build_reference(dataconfigs, reference):
    if reference is not None:
        return reference
    if dataconfigs is None:
        raise ValueError("pass reference=... or dataconfigs=[...] to supply the germline reference "
                         "(it must match the one the model was trained on)")
    import GenAIRR.data as gd
    dcs = [getattr(gd, d) if isinstance(d, str) else d for d in dataconfigs]
    return ReferenceSet.from_dataconfigs(*dcs)


def load_model(checkpoint_path: str, *, dataconfigs=None, reference: ReferenceSet | None = None,
               device: str = "cpu", trust_pickle: bool = False) -> tuple[AlignAIR, ReferenceSet]:
    """Load a trained AlignAIR checkpoint into an eval-ready ``(model, reference)`` pair.

    A safe ``.alignair`` model file carries its own reference (rebuilt from the no-pickle
    ``reference_json``), so ``dataconfigs``/``reference`` are unnecessary and no ``trust_pickle`` is
    needed. A legacy ``.pt`` checkpoint loads via ``torch.load`` (arbitrary-code pickle) and so is
    refused unless ``trust_pickle=True`` — pass it only for a local file you trust, or convert the
    checkpoint first (``alignair convert x.pt x.alignair --dataconfig … --trust-pickle``).
    """
    from .model_file import container, load_model as _load_alignair
    if container.is_alignair_file(checkpoint_path):
        lm = _load_alignair(checkpoint_path, device=device, trust_pickle=trust_pickle)
        return lm.model, (reference or lm.reference)
    if not trust_pickle:
        raise ValueError(
            f"{checkpoint_path} is a legacy .pt checkpoint; loading it runs torch.load "
            "(arbitrary-code pickle). Pass trust_pickle=True only for a local file you trust, or "
            "convert it once: `alignair convert x.pt x.alignair --dataconfig … --trust-pickle`.")
    ck = torch.load(checkpoint_path, map_location=device, weights_only=False)   # legacy .pt (trusted)
    cfg = AlignAIRConfig(**ck["config"])
    model = AlignAIR(cfg).to(device).eval()
    model.load_state_dict(_remap_state_dict(ck["model"]), strict=True)
    return model, _build_reference(dataconfigs, reference)


def predict_sequences(model: AlignAIR, reference: ReferenceSet, sequences: Sequence[str], *,
                      device: str | None = None, batch_size: int = 64, **predict_overrides) -> list:
    """Run the full AlignAIR prediction pipeline on raw nucleotide strings, returning AIRR-convention
    records. ``predict_overrides`` forwards to :class:`~alignair.predict.PredictConfig` (e.g.
    ``threshold=``, ``germline_reader=``, ``genotype=``)."""
    device = device or next(model.parameters()).device.type
    cfg = model.cfg
    pcfg = PredictConfig(max_seq_length=cfg.max_seq_length, has_d=cfg.has_d,
                         batch_size=batch_size, **predict_overrides)
    return _predict(model, list(sequences), reference, pcfg, device=device)


def train_model(dataconfigs, *, out_path: str, steps: int = 100_000, device: str = "cpu",
                **train_overrides) -> str:
    """Train a new AlignAIR model on one or more GenAIRR dataconfigs (1 => single-chain, N =>
    multi-chain) and save a resumable checkpoint to ``out_path``. ``train_overrides`` forwards to
    :func:`alignair.train.trainer.train` (e.g. ``lr=``, ``batch_size=``, ``progresses=``,
    ``short_boost=``, ``resume_path=``)."""
    import GenAIRR.data as gd
    from .core.losses import make_logvars
    from .train.trainer import train as _train

    dcs = [getattr(gd, d) if isinstance(d, str) else d for d in dataconfigs]
    reference = ReferenceSet.from_dataconfigs(*dcs)
    cfg = AlignAIRConfig.from_dataconfigs(*dcs)
    model = AlignAIR(cfg)
    logvars = make_logvars(cfg)
    _train(model, reference, dcs[0], cfg, logvars, steps=steps, device=device,
           save_path=out_path, **train_overrides)
    return out_path
