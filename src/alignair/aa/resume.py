"""Full-state checkpoint I/O for faithful training resume (optimizer/scheduler/RNG/provenance)."""
from __future__ import annotations
import numpy as np, torch

def capture_rng():
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    return (np.random.get_state(), torch.get_rng_state(), cuda_state)

def restore_rng(rng_numpy, rng_torch, rng_cuda=None):
    if rng_numpy is not None:
        np.random.set_state(rng_numpy)
    if rng_torch is not None:
        torch.set_rng_state(rng_torch if isinstance(rng_torch, torch.Tensor) else torch.tensor(rng_torch, dtype=torch.uint8))
    if rng_cuda is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(rng_cuda)

def save_checkpoint(path, *, model, cfg, step, optimizer, scheduler, lr, history, provenance):
    rng_numpy, rng_torch, rng_cuda = capture_rng()
    torch.save({
        "model": model.state_dict(), "cfg": cfg, "step": step,
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "scheduler": scheduler.state_dict() if scheduler is not None else None,
        "rng_numpy": rng_numpy, "rng_torch": rng_torch, "rng_cuda": rng_cuda,
        "lr": lr, "history": history, "provenance": provenance,
    }, path)

def load_checkpoint(path) -> dict:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    return {k: ck.get(k) for k in
            ("model", "cfg", "step", "optimizer", "scheduler", "rng_numpy", "rng_torch", "rng_cuda",
             "lr", "history", "provenance")}
