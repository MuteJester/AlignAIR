"""Generate labelled GenAIRR reads, run a model, score per stratum."""
from __future__ import annotations

import itertools
import math
import statistics

from ..api import predict_sequences


def default_strata() -> dict:
    """Named GenAIRR generation params: clean, moderate corruption, heavy-SHM, and a J-anchored short
    amplicon. (The full 22-stratum head-to-head lives in the ``alignair_benchmark`` package.)"""
    from ..train.gym import Curriculum
    cur = Curriculum()
    return {
        "clean": dict(cur.params(0.1)),
        "moderate": dict(cur.params(0.6)),
        "high_shm": {**cur.params(0.3), "mutation_rate": 0.25},
        "short_janchor": {**cur.params(0.6), "end_loss_5": (150, 350), "end_loss_3": (0, 15)},
    }


def generate_labeled(dataconfig, params: dict, n: int, seed: int) -> list[dict]:
    """``n`` labelled records (truth calls + coords + junction) from the GenAIRR gym stream."""
    from ..train.gym import build_experiment
    p = dict(params)
    p.setdefault("invert_d_prob", 0.0)
    exp = build_experiment(dataconfig, p, allow_curatable=True)
    return list(itertools.islice(exp.stream_records(n=None, seed=seed), n))


def _first(call) -> str:
    return str(call).split(",")[0].strip() if call else ""


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _call_correct(pred_call, truth_call) -> bool:
    pf = _first(pred_call)
    truth_set = {c.strip() for c in str(truth_call or "").split(",") if c.strip()}
    return bool(pf) and pf in truth_set


def _mean(xs):
    return statistics.mean(xs) if xs else math.nan


def score(truth: list[dict], preds: list[dict]) -> dict:
    """Per-metric scores of ``preds`` against ``truth`` (aligned lists)."""
    m: dict = {}
    for g in ("v", "d", "j"):
        pairs = [(t, p) for t, p in zip(truth, preds) if t.get(f"{g}_call")]
        m[f"{g}_call_acc"] = _mean([_call_correct(p.get(f"{g}_call"), t.get(f"{g}_call")) for t, p in pairs])
    jt = [(t, p) for t, p in zip(truth, preds) if t.get("junction")]
    m["junction_nt_exact"] = _mean([(p.get("junction") or "").upper() == str(t["junction"]).upper()
                                    for t, p in jt])
    # length MAE is a softer signal than exact-match: an exact-match of 0 with a small length MAE
    # says the junction is only a few nt off (boundary jitter), not structurally wrong.
    m["junction_len_mae"] = _mean([abs(len(p.get("junction") or "") - len(str(t["junction"]))) for t, p in jt])
    for g in ("v", "j"):
        for b in ("sequence_start", "sequence_end"):
            errs = [abs(_num(p.get(f"{g}_{b}")) - _num(t.get(f"{g}_{b}")))
                    for t, p in zip(truth, preds)
                    if _num(p.get(f"{g}_{b}")) is not None and _num(t.get(f"{g}_{b}")) is not None]
            m[f"{g}_{b}_mae"] = _mean(errs)
    m["productive_acc"] = _mean([bool(p.get("productive")) == bool(t.get("productive"))
                                 for t, p in zip(truth, preds)])
    return m


def run_benchmark(model, reference, dataconfig, *, n: int = 200, seed: int = 0,
                  strata_names: list[str] | None = None, device: str = "cpu", batch_size: int = 64) -> dict:
    strata = default_strata()
    if strata_names:
        strata = {k: strata[k] for k in strata_names if k in strata}
    out: dict = {}
    for i, (name, params) in enumerate(strata.items()):
        truth = generate_labeled(dataconfig, params, n, seed + i)
        preds = predict_sequences(model, reference, [r["sequence"] for r in truth],
                                  device=device, batch_size=batch_size)
        out[name] = {"n": len(truth), **score(truth, preds)}
    return out


def format_text(results: dict) -> str:
    metrics = ["v_call_acc", "d_call_acc", "j_call_acc", "junction_nt_exact", "junction_len_mae",
               "v_sequence_start_mae", "v_sequence_end_mae", "j_sequence_start_mae",
               "j_sequence_end_mae", "productive_acc"]
    metrics = [k for k in metrics if any(k in m for m in results.values())]
    L = ["AlignAIR benchmark (self-eval on generated GenAIRR reads)", "=" * 60]
    def _abbr(k):
        return (k.replace("_call_acc", "").replace("sequence_", "")
                .replace("junction_nt_exact", "junc_nt").replace("junction_len_mae", "junc_lenMAE"))
    head = f"{'stratum':16s} {'n':>5s} " + " ".join(f"{_abbr(k):>11s}" for k in metrics)
    L.append(head)
    for name, m in results.items():
        cells = " ".join(f"{m.get(k, float('nan')):>11.3f}" for k in metrics)
        L.append(f"{name:16s} {m.get('n', 0):>5d} {cells}")
    return "\n".join(L)
