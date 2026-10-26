"""Single-run guard optimization with explicit study configuration."""
from __future__ import annotations

import hashlib
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from alignair.aa.guard.checkpoint import build_provenance
from alignair.aa.guard.features import FEATURE_NAMES
from alignair.aa.guard.fusion import fit_fusion
from alignair.aa.guard.losses import energy_margin_loss, supcon_loss
from alignair.aa.guard.model import GuardModel
from alignair.aa.guard.tokenizer import MAX_LEN, AaTokenizer
from alignair.errors import AlignAIRError, ArtifactIntegrityError, ConfigurationError, NumericalError, TrainingError
from . import budget
from .data import build_tensors
from .metrics import binary_auroc


@dataclass(frozen=True)
class RunConfig:
    """Study settings and output paths supplied by the caller."""

    manifest: dict
    checkpoint_dir: str | Path
    budget_path: str | Path
    device: str = "cpu"


def resolve_device(requested):
    """Resolve a CPU/CUDA target when a study is invoked, never during import."""
    if requested == "auto":
        requested = "cuda" if torch.cuda.is_available() else "cpu"
    try:
        device = torch.device(requested)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ConfigurationError("Invalid guard training device.", context={"device": requested}) from exc
    if device.type not in ("cpu", "cuda"):
        raise ConfigurationError("Guard training supports CPU and CUDA devices.", context={"device": str(device)})
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ConfigurationError("CUDA was requested but is unavailable.", context={"device": str(device)})
    return str(device)


def _run_contract(cid, seed, train, dev, oe, feat_cache, cat2id, eligible_dt, n_sources, config,
                  mask_groups, extra_provenance):
    """Bind completed and resumable checkpoints to their exact training inputs."""
    inputs = {"train": train, "dev": dev, "oe": oe, "features": feat_cache}
    contract = {
        "schema_version": 1, "candidate": cid, "seed": int(seed),
        "manifest": config.manifest, "source_vocabulary": cat2id, "n_sources": n_sources,
        "eligible_domain_types": sorted(eligible_dt), "masked_profiles": sorted(mask_groups),
        "extra_provenance": extra_provenance or {},
        "inputs_sha256": hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest(),
    }
    return hashlib.sha256(json.dumps(contract, sort_keys=True).encode()).hexdigest()


def _verify_contract(saved, expected, path):
    if saved.get("training_contract") != expected:
        raise ArtifactIntegrityError(
            "Guard checkpoint does not match the requested training run.",
            context={"path": path},
            hint="Use a new output directory for a different run. Legacy unbound checkpoints require explicit migration.",
        )


def _forward_losses(model, T, W, margins, cfg, grl, device):
    out = model(T["ids"].to(device), T["mask"].to(device),
                hmm_norm=T["hmm"].to(device) if cfg["fusion"] else None, grl_lambda=grl)
    dev = device
    parts = {}
    tm = T["type_m"].to(dev)
    parts["type"] = (F.cross_entropy(out["domain_type"], T["domain_type"].to(dev), reduction="none") * tm
                     ).sum() / tm.sum().clamp(min=1)
    cm = T["comp_m"].to(dev)
    parts["complete"] = (F.cross_entropy(out["completeness"], T["completeness"].to(dev), reduction="none") * cm
                         ).sum() / cm.sum().clamp(min=1)
    parts["app"] = F.binary_cross_entropy_with_logits(out["applicability"].squeeze(-1), T["applicability"].to(dev))
    parts["energy"] = energy_margin_loss(out["energy"], T["energy_out"].to(dev), T["energy_m"].to(dev),
                                         m_in=margins["m_in"], m_out=margins["m_out"]) if cfg["energy"] \
        else out["energy"].sum() * 0.0
    parts["contrast"] = supcon_loss(out["rep"], T["contrast"].to(dev), T["contrast_m"].to(dev)) if cfg["contrast"] \
        else out["rep"].sum() * 0.0
    if cfg["source"]:
        sm = T["source_m"].to(dev)
        parts["source"] = (F.cross_entropy(out["source_logits"], T["source"].to(dev), reduction="none") * sm
                           ).sum() / sm.sum().clamp(min=1) if sm.sum() > 0 else out["source_logits"].sum() * 0.0
    else:
        parts["source"] = out["source_logits"].sum() * 0.0
    total = (parts["type"] + W["complete"] * parts["complete"] + W["app"] * parts["app"]
             + W["oe"] * parts["energy"] + W["contrast"] * parts["contrast"] + W["source"] * parts["source"])
    return total, parts, out


def _rng_state(device):
    return {"torch": torch.get_rng_state(), "cuda": torch.cuda.get_rng_state_all() if torch.device(device).type == "cuda" else None,
            "python": random.getstate(), "numpy": np.random.get_state()}


def train_one(cid, seed, train, dev, oe, feat_cache, cat2id, eligible_dt, n_sources, *,
              config: RunConfig, bank, mask_groups=frozenset(), ckpt_dir=None, extra_provenance=None, run_tag=None):
    manifest = config.manifest
    device = resolve_device(config.device)
    is_cuda = torch.device(device).type == "cuda"
    from alignair.aa.guard.model import CANDIDATE_CONFIG
    if cid not in CANDIDATE_CONFIG:
        raise ConfigurationError("Unknown guard training candidate.", context={"candidate": cid})
    if not train or not dev:
        raise ConfigurationError("Guard training requires nonempty training and development records.")
    if bank.checksum() != manifest["bank_checksum"]:
        raise ArtifactIntegrityError("Profile bank differs from training manifest.")
    ckpt_dir = Path(ckpt_dir) if ckpt_dir else Path(config.checkpoint_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = run_tag or f"{cid}.s{seed}"                  # LOCO passes e.g. C1.s2.igk_igl (no filename collision)
    final_path = ckpt_dir / f"{tag}.final.pt"
    contract = _run_contract(cid, seed, train, dev, oe, feat_cache, cat2id, eligible_dt, n_sources,
                             config, mask_groups, extra_provenance)
    if final_path.exists():                            # already completed -> skip on resumable restart
        saved = torch.load(final_path, map_location="cpu", weights_only=False)
        _verify_contract(saved, contract, final_path)
        metrics = saved["metrics"]
        budget.record(tag, metrics["wall_s"], metrics["gpu_h"], metrics["params"], path=config.budget_path)
        return metrics
    ckpt_path = ckpt_dir / f"{tag}.pt"
    cfg = CANDIDATE_CONFIG[cid]                         # static config (no throwaway model / no source_head)
    torch.manual_seed(seed); random.seed(seed); np.random.seed(seed)
    tr = manifest["training"]; W = manifest["loss_weights"]; margins = manifest["energy_margins"]
    bank_checksum = manifest["bank_checksum"]

    fusion = fit_fusion(FEATURE_NAMES, [feat_cache[hashlib.sha256(r["aa"].encode()).hexdigest()]
                                        for r in train], bank_checksum)   # TRAIN-only normalization
    tok = AaTokenizer(max_len=MAX_LEN)
    T_dev = build_tensors(dev, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt, mask_groups=mask_groups, bank=bank)
    T_oe = build_tensors(oe, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt, is_oe=True,
                         mask_groups=mask_groups, bank=bank) if oe else None

    model = GuardModel(cid, hmm_dim=len(FEATURE_NAMES), n_sources=n_sources).to(device)   # size source_head to data
    opt = torch.optim.AdamW(model.parameters(), lr=manifest["optimizer"]["lr"],
                            weight_decay=manifest["optimizer"]["weight_decay"])
    max_steps, warm = tr["max_steps"], manifest["schedule"]["warmup_steps"]
    if max_steps < tr["eval_every"] or tr["eval_every"] <= 0 or warm < 0:
        raise ConfigurationError("Guard training requires a positive evaluation interval within the run budget.")
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / warm if s < warm else 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, max_steps - warm))))
    amp_enabled = is_cuda and tr["amp"]
    if hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    else:  # PyTorch 2.2 support
        scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)
    start_step, best_dev, patience, best_state = 0, float("inf"), 0, None

    resumed = None
    if ckpt_path.exists():                               # resume operationally
        st = torch.load(ckpt_path, map_location=device, weights_only=False)
        _verify_contract(st, contract, ckpt_path)
        model.load_state_dict(st["model"]); opt.load_state_dict(st["opt"]); sched.load_state_dict(st["sched"])
        scaler.load_state_dict(st["scaler"]); start_step = st["step"]; best_dev = st["best_dev"]
        patience = st["patience"]; best_state = st.get("best_state")
        random.setstate(st["rng"]["python"]); np.random.set_state(st["rng"]["numpy"])
        torch.set_rng_state(st["rng"]["torch"].cpu())
        if is_cuda and st["rng"]["cuda"] is not None:
            torch.cuda.set_rng_state_all([state.cpu() for state in st["rng"]["cuda"]])
        resumed = st

    if is_cuda:
        torch.cuda.reset_peak_memory_stats()
    rng = random.Random(seed + 1000)
    if resumed is not None:
        rng.setstate(resumed["sampling_rng"])
    by_role = {"positive": [r for r in train if r["group"] in ("positive_real", "positive_sim")],
               "biological_negative": [r for r in train if r["group"] in ("vhh_nonhuman", "igsf", "igk_igl", "tcr")],
               "structural": [r for r in train if r["group"] == "structural"]}
    roles = [k for k in by_role if by_role[k]]
    if not roles:
        raise ConfigurationError("Guard training requires records with recognized training roles.")
    bs = tr["batch_size"]; n_oe = int(bs * manifest["oe"]["cap_fraction"]) if oe else 0
    per_role = max(1, (bs - n_oe) // len(roles))
    t0 = time.time(); stop_step = resumed.get("best_step", start_step) if resumed else start_step
    train_loss_ema = resumed.get("train_loss_ema") if resumed else None
    progress_step = start_step
    try:
        model.train()
        end_step = max_steps if patience < tr["early_stop_patience"] else start_step
        for step in range(start_step, end_step):
            progress_step = step
            batch = [rng.choice(by_role[r]) for r in roles for _ in range(per_role)]
            Tb = build_tensors(batch, feat_cache, fusion, tok, bank_checksum, cat2id, eligible_dt,
                               mask_groups=mask_groups, bank=bank)
            if oe and n_oe:
                Toe = build_tensors([rng.choice(oe) for _ in range(n_oe)], feat_cache, fusion, tok,
                                    bank_checksum, cat2id, eligible_dt, is_oe=True, mask_groups=mask_groups, bank=bank)
                for k in Tb:
                    Tb[k] = torch.cat([Tb[k], Toe[k]], dim=0)
            opt.zero_grad()
            with torch.autocast(device_type=torch.device(device).type, enabled=amp_enabled):
                total, parts, _ = _forward_losses(model, Tb, W, margins, cfg, manifest["grl_lambda"], device)
            if not torch.isfinite(total):
                raise NumericalError("Non-finite guard training loss.", context={"run_tag": tag, "step": step})
            scaler.scale(total).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), tr["grad_clip"], error_if_nonfinite=True)
            scaler.step(opt); scaler.update(); sched.step()
            train_loss = float(total.detach())
            train_loss_ema = train_loss if train_loss_ema is None else 0.9 * train_loss_ema + 0.1 * train_loss

            if (step + 1) % tr["eval_every"] == 0:
                model.eval()
                with torch.no_grad():
                    dtot, dparts, dout = _forward_losses(model, T_dev, W, margins, cfg, manifest["grl_lambda"], device)
                    if T_oe is not None:
                        _, _, ooe = _forward_losses(model, T_oe, W, margins, cfg, manifest["grl_lambda"], device)
                dv = float(dtot)
                if not math.isfinite(dv):
                    raise NumericalError("Non-finite guard development loss.", context={"run_tag": tag, "step": step})
                if dv < best_dev - tr["min_delta"]:
                    best_dev, patience, stop_step = dv, 0, step + 1
                    best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                else:
                    patience += 1
                torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(),
                            "scaler": scaler.state_dict(), "step": step + 1, "best_dev": best_dev,
                            "patience": patience, "best_state": best_state, "rng": _rng_state(device),
                            "sampling_rng": rng.getstate(), "train_loss_ema": train_loss_ema,
                            "training_contract": contract, "best_step": stop_step}, ckpt_path)
                model.train()
                if patience >= tr["early_stop_patience"]:
                    break
    except (torch.cuda.OutOfMemoryError, FloatingPointError, RuntimeError, NumericalError) as e:
        error = e if isinstance(e, AlignAIRError) else TrainingError(
            "Guard training run failed.", context={"run_tag": tag, "step": progress_step, "cause_type": type(e).__name__})
        failure = {"tag": tag, "cid": cid, "seed": seed, "status": "engineering_failure",
                   "reason": f"{type(e).__name__}: {str(e)[:120]}", "step": progress_step, "error": error.to_dict()}
        (ckpt_dir / f"{tag}.failure.json").write_text(json.dumps(failure, indent=2))
        return failure

    if best_state is not None:
        model.load_state_dict(best_state)
    metrics = _evaluate(model, T_dev, T_oe, cfg, W, margins, device)
    metrics.update({"tag": tag, "cid": cid, "seed": seed, "status": "ok", "stop_step": stop_step,
                    "dev_total_loss": best_dev, "train_loss_ema": round(train_loss_ema or 0.0, 4),
                    "train_dev_gap": round((train_loss_ema or 0.0) - best_dev, 4),
                    "wall_s": round(time.time() - t0, 1),
                    "gpu_h": round((time.time() - t0) / 3600, 4) if is_cuda else 0.0,
                    "peak_vram_mb": round(torch.cuda.max_memory_allocated() / 1e6, 1) if is_cuda else 0,
                    "params": sum(p.numel() for p in model.parameters()),
                    "eligible_source_fraction": round(float(T_dev["source_m"].mean()), 4)})
    xp = dict(extra_provenance or {})
    xp["normalization_stats_sha256"] = hashlib.sha256(fusion.mean.tobytes() + fusion.std.tobytes()).hexdigest()
    prov = build_provenance(bank_checksum=bank_checksum,
                            training_manifest_hash=xp.get("training_manifest_hash", manifest["manifest_sha256"]),
                            partition_hash=xp.get("partition_hash", manifest["train_partition_hash"]), seed=seed,
                            loss_mask_config={"candidate": cid, "weights": W, "margins": margins,
                                              "loco_mask_hash": xp.get("loco_mask_hash"), "fold": xp.get("fold"),
                                              "norm_stats_sha256": xp["normalization_stats_sha256"]})
    torch.save({"model": model.state_dict(), "provenance": prov.to_dict(), "metrics": metrics,
                "bound_extra": xp, "training_contract": contract},
               final_path)
    budget.record(tag, metrics["wall_s"], metrics["gpu_h"], metrics["params"], path=config.budget_path)
    return metrics


def _evaluate(model, T_dev, T_oe, cfg, W, margins, device):
    model.eval()
    with torch.no_grad():
        _, parts, out = _forward_losses(model, T_dev, W, margins, cfg, 1.0, device)
        app_logit = out["applicability"].squeeze(-1).cpu().numpy()
        app_y = T_dev["applicability"].numpy()
        dt_pred = out["domain_type"].argmax(-1).cpu().numpy(); dt_y = T_dev["domain_type"].numpy()
        dt_m = T_dev["type_m"].numpy().astype(bool)
        cp_pred = out["completeness"].argmax(-1).cpu().numpy(); cp_y = T_dev["completeness"].numpy()
        cp_m = T_dev["comp_m"].numpy().astype(bool)
        src_pred = out["source_logits"].argmax(-1).cpu().numpy(); src_y = T_dev["source"].numpy()
        src_m = T_dev["source_m"].numpy().astype(bool)
        e_in = out["energy"].cpu().numpy()
        e_oe = None
        if T_oe is not None:
            _, _, ooe = _forward_losses(model, T_oe, W, margins, cfg, 1.0, device)
            e_oe = ooe["energy"].cpu().numpy()

    def _acc(pred, y, m):
        return round(float((pred[m] == y[m]).mean()), 4) if m.any() else None

    pos, neg = app_y == 1, app_y == 0
    return {
        "component_losses": {k: round(float(v), 4) for k, v in parts.items()},
        "type_acc": _acc(dt_pred, dt_y, dt_m), "completeness_acc": _acc(cp_pred, cp_y, cp_m),
        "applicability_auroc": binary_auroc(app_y, app_logit),
        "pos_recall": round(float(((app_logit > 0) & pos).sum() / max(1, pos.sum())), 4),
        "neg_recall": round(float(((app_logit <= 0) & neg).sum() / max(1, neg.sum())), 4),
        "source_acc": _acc(src_pred, src_y, src_m),
        "energy_in_mean": round(float(e_in.mean()), 3),
        "energy_oe_mean": round(float(e_oe.mean()), 3) if e_oe is not None else None,
        "energy_separation": round(float(e_oe.mean() - e_in.mean()), 3) if e_oe is not None else None,
    }
