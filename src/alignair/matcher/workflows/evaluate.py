"""CLI: evaluate AIRRMatch set-aware allele accuracy on a (held-out) dataconfig."""
import argparse

import torch
from alignair.core.runtime import restore_model_state
from alignair.errors import ModelFormatError

from alignair.matcher.config import MatcherConfig
from alignair.matcher.model import AIRRMatch
from alignair.matcher.checkpoint import read_checkpoint
from alignair.matcher.evaluate import evaluate  # re-exported for `from eval_matcher import evaluate`


def _resolve(name):
    import GenAIRR.data as gd
    return getattr(gd, name)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--config", default="HUMAN_IGH_OGRDB", help="held-out dataconfig name")
    ap.add_argument("-n", type=int, default=500)
    ap.add_argument("--progress", type=float, default=0.3)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()
    ck = read_checkpoint(a.model)
    try:
        cfg = MatcherConfig(**ck["config"])
    except TypeError as exc:
        raise ModelFormatError("matcher checkpoint has invalid configuration fields") from exc
    model = AIRRMatch(cfg)
    restore_model_state(model, ck["model"], source=a.model)
    acc = evaluate(model, _resolve(a.config), cfg, n=a.n, progress=a.progress, device=a.device)
    print("set-aware allele accuracy:", {g: round(v, 3) for g, v in acc.items()})
