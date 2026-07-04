"""Full-state checkpoints so training resumes seamlessly (weights + optimizer + step)."""
from dataclasses import asdict, is_dataclass

import torch


def save_checkpoint(path, cfg, model, opt, step):
    cfg_d = asdict(cfg) if is_dataclass(cfg) else dict(vars(cfg))
    torch.save({"config": cfg_d, "model": model.state_dict(),
                "opt": opt.state_dict(), "step": int(step)}, path)


def load_checkpoint(path, model, opt=None, device="cpu"):
    """Load weights (always) and optimizer state (if opt given AND present). Returns step.

    Passing opt=None gives a weights-only warm-start (step treated as 0 by the caller);
    passing an optimizer restores momentum/variance for a seamless resume.
    """
    ck = torch.load(path, map_location=device)
    model.load_state_dict(ck["model"])
    if opt is not None and ck.get("opt") is not None:
        opt.load_state_dict(ck["opt"])
    return int(ck.get("step", 0))
