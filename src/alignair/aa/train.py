"""Phase-1 training helpers: overfit-one-batch capacity proof + per-task metrics."""
from __future__ import annotations

import torch

from .loss import aa_loss


@torch.no_grad()
def aa_eval_metrics(out: dict, targets: dict, genes=("v", "j")) -> dict:
    m = {}
    for g in genes:
        pred = out[f"{g}_class_logits"].argmax(-1)
        m[f"{g}_class_acc"] = float((pred == targets[f"{g}_class"]).float().mean())
        for b in ("start", "end"):
            m[f"{g}_{b}_mae"] = float(
                (out[f"{g}_{b}"].squeeze(-1) - targets[f"{g}_{b}"].squeeze(-1)).abs().mean())
    return m


def assert_batch_diverse(samples, min_v_classes: int = 3, min_j_classes: int = 2) -> None:
    """A memorization proof on a single-class batch proves nothing — require real diversity."""
    nv = len({s["v_class"] for s in samples})
    nj = len({s["j_class"] for s in samples})
    if nv < min_v_classes or nj < min_j_classes:
        raise ValueError(f"batch not diverse enough: {nv} V classes (need >= {min_v_classes}), "
                         f"{nj} J classes (need >= {min_j_classes})")


def diverse_selector(min_v_classes: int = 3, min_j_classes: int = 2):
    """A ``select`` callable for ``collect_usable`` that ACTIVELY builds a diverse batch.

    Takes a sample only if it contributes an unseen V or J class, until the diversity quota is
    met; afterwards it accepts freely to fill the batch. Beats taking the first N records and
    hoping ``assert_batch_diverse`` passes."""
    def select(sample, chosen):
        seen_v = {c["v_class"] for c in chosen}
        seen_j = {c["j_class"] for c in chosen}
        if len(seen_v) >= min_v_classes and len(seen_j) >= min_j_classes:
            return True                                   # quota met -> fill the rest
        return sample["v_class"] not in seen_v or sample["j_class"] not in seen_j
    return select


def overfit_one_batch(model, batch_in, targets, *, steps=300, lr=1e-3, genes=("v", "j")) -> dict:
    """Drive one batch to memorization. Returns initial/final loss, final metrics, and the
    per-step component history."""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    history, initial = [], None
    for _ in range(steps):
        out = model(batch_in)
        total, parts = aa_loss(out, targets, genes=genes)
        history.append({"total": float(total.detach()),
                        **{k: float(v.detach()) for k, v in parts.items()}})
        if initial is None:
            initial = float(total.detach())
        opt.zero_grad()
        total.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        out = model(batch_in)
        final = float(aa_loss(out, targets, genes=genes)[0])
        metrics = aa_eval_metrics(out, targets, genes)
    return {"initial_loss": initial, "final_loss": final, "history": history, **metrics}
