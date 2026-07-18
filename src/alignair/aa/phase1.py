"""Phase-1 learning proof: per-seed capacity (overfit) + generalization on ONE frozen held-out
set evaluated against baselines. Freezes full provenance so a class id is interpretable later."""
from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
from collections import Counter

import torch

from .config import aa_igh_config
from .data import build_class_maps, check_counter, collate_aa, collect_usable
from .evaluate import evaluate_frozen
from .loss import aa_loss
from .model import AAAlignAIR
from .tokenizer import AAProteinTokenizer
from .train import assert_batch_diverse, diverse_selector, overfit_one_batch

# Disjoint seed ranges. GenAIRR consumes CONSECUTIVE per-record seeds, so a stream seeded S
# occupies [S, S+generated). A 3000x64 run needs ~192k usable ~= ~1.4M raw records at the observed
# ~14% yield, so training ranges are 50M apart with a 20M budget -- provably non-overlapping.
SEED_RANGES = {
    "heldout": {"start": 1_000_000_000, "budget": 5_000_000},
    "overfit": {"start": 2_000_000_000, "stride": 50_000_000, "budget": 5_000_000},
    "train": {"start": 3_000_000_000, "stride": 50_000_000, "budget": 20_000_000},
}


def _range_for(kind: str, replicate: int = 0) -> dict:
    r = SEED_RANGES[kind]
    start = r["start"] + replicate * r.get("stride", 0)
    return {"start_seed": start, "budget": r["budget"]}


def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:                                    # noqa: BLE001 - provenance must not break a run
        return "unknown"


def heldout_fingerprint(samples) -> str:
    """SHA-256 over a canonical serialization of the held-out cohort. 'Frozen' must be verifiable,
    not merely 'reproducible from a seed'."""
    h = hashlib.sha256()
    for s in samples:
        h.update(",".join(str(int(t)) for t in s["tokens"]).encode())
        h.update(f"|{s['v_class']},{s['j_class']},{s['v_start']},{s['v_end']},"
                 f"{s['j_start']},{s['j_end']},{s['v_end'] - s['v_start']}\n".encode())
    return h.hexdigest()


def write_heldout_manifest(samples, path: str) -> str:
    import gzip
    with gzip.open(path, "wt") as f:
        for s in samples:
            f.write(json.dumps({"tokens": [int(t) for t in s["tokens"]],
                                "v_class": s["v_class"], "j_class": s["j_class"],
                                "v_start": s["v_start"], "v_end": s["v_end"],
                                "j_start": s["j_start"], "j_end": s["j_end"]},
                               sort_keys=True) + "\n")
    return path


def provenance(cfg, class_maps, meta, model) -> dict:
    import GenAIRR
    maps = {g: class_maps[g].to_json() for g in ("V", "J")}
    sha = {g: hashlib.sha256(json.dumps(maps[g], sort_keys=True).encode()).hexdigest()
           for g in ("V", "J")}
    return {
        "class_maps": maps,
        "class_map_sha256": sha,
        "db_hash": meta["db_hash"],
        "config": dataclasses.asdict(cfg),
        "versions": {"genairr": getattr(GenAIRR, "__version__", "unknown"),
                     "torch": torch.__version__,
                     "cuda": torch.version.cuda,
                     "cudnn": (torch.backends.cudnn.version() if torch.cuda.is_available() else None)},
        "git_commit": _git_commit(),
        "param_count": sum(p.numel() for p in model.parameters()),
        "quarantined_v": meta["quarantined_v"], "quarantined_j": meta["quarantined_j"],
    }


def run_seed(seed, cfg, class_maps, heldout, *, replicate, train_steps, batch_size, lr,
             overfit_batch, overfit_steps, device, dc_name) -> dict:
    tok = AAProteinTokenizer()
    counters = {"overfit": Counter(), "train": Counter()}

    # (a) capacity: overfit one ACTIVELY DIVERSE batch, from this replicate's own seed range
    torch.manual_seed(seed)
    ob_range = _range_for("overfit", replicate)
    ob, ob_info = collect_usable(**ob_range, want=overfit_batch, class_maps=class_maps,
                                 tokenizer=tok, max_len=cfg.max_seq_length, dc_name=dc_name,
                                 counter=counters["overfit"], select=diverse_selector())
    assert_batch_diverse(ob)
    obi, obt = collate_aa(ob, cfg.max_seq_length, device)
    overfit = overfit_one_batch(AAAlignAIR(cfg).to(device), obi, obt,
                                steps=overfit_steps, lr=lr)

    # (b) generalization. RE-SEED first so this model's init does not depend on how much RNG the
    # overfit run above consumed.
    torch.manual_seed(seed + 10_000)
    tr_range = _range_for("train", replicate)
    train_samples, tr_info = collect_usable(**tr_range, want=train_steps * batch_size,
                                            class_maps=class_maps, tokenizer=tok,
                                            max_len=cfg.max_seq_length, dc_name=dc_name,
                                            counter=counters["train"])
    model = AAAlignAIR(cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    model.train()
    losses = []
    for i in range(train_steps):
        batch = train_samples[i * batch_size:(i + 1) * batch_size]
        bi, bt = collate_aa(batch, cfg.max_seq_length, device)
        total, parts = aa_loss(model(bi), bt)
        losses.append({"total": float(total.detach()),
                       **{k: float(v.detach()) for k, v in parts.items()}})
        opt.zero_grad()
        total.backward()
        opt.step()

    for c in counters.values():
        check_counter(c)                       # generated == usable + rejections, per stream

    return {"overfit": {k: v for k, v in overfit.items() if k != "history"},
            "overfit_history_last": overfit["history"][-1],
            "train_loss_first": losses[0] if losses else None,
            "train_loss_last": losses[-1] if losses else None,
            "seed_ranges": {"overfit": ob_info, "train": tr_info},
            "counters": {k: dict(v) for k, v in counters.items()},
            "heldout": evaluate_frozen(model, heldout, cfg, class_maps,
                                       batch_size=batch_size, device=device)}


def run_learning_proof(*, seeds=(0, 1, 2), heldout_size=4096, train_steps=3000, batch_size=64,
                       lr=1e-3, overfit_batch=32, overfit_steps=400, device="cpu",
                       dc_name="HUMAN_IGH_OGRDB", manifest_path=None) -> dict:
    class_maps, meta = build_class_maps(dc_name)
    cfg = aa_igh_config(meta["n_v_classes"], meta["n_j_classes"])
    tok = AAProteinTokenizer()

    # ONE held-out cohort from its own seed range, disjoint from every train/overfit range.
    ho_counter = Counter()
    heldout, ho_info = collect_usable(**_range_for("heldout"), want=heldout_size,
                                      class_maps=class_maps, tokenizer=tok,
                                      max_len=cfg.max_seq_length, dc_name=dc_name,
                                      counter=ho_counter)
    check_counter(ho_counter)

    per_seed = {}
    for rep, sd in enumerate(seeds):
        per_seed[str(sd)] = run_seed(sd, cfg, class_maps, heldout, replicate=rep,
                                     train_steps=train_steps, batch_size=batch_size, lr=lr,
                                     overfit_batch=overfit_batch, overfit_steps=overfit_steps,
                                     device=device, dc_name=dc_name)

    ref_model = AAAlignAIR(cfg)
    out = {"schema": "alignair.aa.phase1.v2",
           "provenance": provenance(cfg, class_maps, meta, ref_model),
           "heldout_size": len(heldout),
           "heldout_fingerprint": heldout_fingerprint(heldout),
           "heldout_seed_range": ho_info,
           "heldout_counters": dict(ho_counter),
           "seed_ranges_declared": SEED_RANGES,
           "settings": {"seeds": list(seeds), "train_steps": train_steps,
                        "batch_size": batch_size, "lr": lr, "overfit_batch": overfit_batch,
                        "overfit_steps": overfit_steps, "device": device},
           "seeds": per_seed}
    if manifest_path:
        out["heldout_manifest"] = write_heldout_manifest(heldout, manifest_path)
    return out
