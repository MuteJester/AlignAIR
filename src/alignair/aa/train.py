"""Phase-1 training helpers: overfit-one-batch capacity proof + per-task metrics."""
from __future__ import annotations

import torch

from ..tasks import build_tasks
from ..train.setup import build_training_engine
from ..tasks.metrics import categorical_metrics as aa_eval_metrics


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
    tasks = build_tasks(model.cfg, genes=genes)
    engine = build_training_engine(model, tasks=tasks, lr=lr)
    history, initial = [], None
    for step in range(1, steps + 1):
        result = engine.step((batch_in, targets), step=step)
        history.append({"total": result.total, **result.parts})
        if initial is None:
            initial = result.total
    model.eval()
    with torch.no_grad():
        out = model(batch_in)
        final = float(tasks.loss(out, targets)[0])
        metrics = aa_eval_metrics(out, targets, genes)
    return {"initial_loss": initial, "final_loss": final, "history": history, **metrics}
