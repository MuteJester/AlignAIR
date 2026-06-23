#!/usr/bin/env python
"""Measure AlignAIR prediction throughput + training step time and emit a Markdown table.

Speed depends on the model architecture, sequence length, and reference size — NOT on whether
the weights are trained — so this times representative architectures with untrained weights.
Reproduce with:  PYTHONPATH=src .venv/bin/python scripts/perf_table.py [N_READS]

Writes docs/performance.md.
"""
import argparse
import gc
import platform
import statistics
import time

import torch

import GenAIRR.data as gdata
from alignair.api import LoadedModel, predict
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR
from alignair.reference.reference_set import ReferenceSet
from alignair.gym.gym import build_experiment
from alignair.gym.curriculum import Curriculum

PRESETS = {"desktop": (128, 4, 8), "standard": (256, 8, 8)}


def make_model(preset, device):
    d_model, n_layers, nhead = PRESETS[preset]
    torch.manual_seed(0)
    m = DNAlignAIR(DNAlignAIRConfig(d_model=d_model, n_layers=n_layers, nhead=nhead)).to(device)
    m.eval()
    return m


def gen_reads(n, seed=1):
    exp = build_experiment(gdata.HUMAN_IGH_OGRDB, Curriculum().params(0.3))
    return [r["sequence"] for r in exp.stream_records(n=n, seed=seed)]


def peak_mem(device, fn):
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
        fn()
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e9
    import psutil
    fn()
    return psutil.Process().memory_info().rss / 1e9


def time_predict(loaded, reads, *, full_alignment, v_reader, batch=64, passes=3):
    predict(loaded, reads[: min(32, len(reads))], full_alignment=full_alignment,
            v_reader=v_reader, batch_size=batch)                              # warm up
    rates = []
    for _ in range(passes):
        if loaded.device == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        predict(loaded, reads, full_alignment=full_alignment, v_reader=v_reader, batch_size=batch)
        if loaded.device == "cuda":
            torch.cuda.synchronize()
        rates.append(len(reads) / (time.time() - t0))
    return statistics.median(rates)                                          # robust to noise


def time_train(preset, device, steps=6):
    from alignair.losses.dnalignair_loss import DNAlignAIRLoss
    from alignair.gym.gym import AlignAIRGym
    from alignair.training.gym_trainer import GymTrainer
    dc = gdata.HUMAN_IGH_OGRDB
    rs = ReferenceSet.from_dataconfigs(dc)
    model = make_model(preset, device)
    gym = AlignAIRGym([dc], rs, seed=0)
    trainer = GymTrainer(model, DNAlignAIRLoss(has_d=rs.has_d), rs, gym, lr=5e-4, batch_size=32)
    trainer.fit(total_steps=1, global_total=steps, progress=False)            # warm up
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats(); torch.cuda.synchronize()
    t0 = time.time()
    trainer.fit(total_steps=steps, global_total=steps, progress=False)
    if device == "cuda":
        torch.cuda.synchronize()
    s_step = (time.time() - t0) / steps
    mem = (torch.cuda.max_memory_allocated() / 1e9 if device == "cuda"
           else __import__("psutil").Process().memory_info().rss / 1e9)
    return s_step, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("n_reads", nargs="?", type=int, default=300)
    args = ap.parse_args()

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    rs_full = ReferenceSet.from_dataconfigs(gdata.HUMAN_IGH_OGRDB)
    reads = gen_reads(args.n_reads)
    avg_len = round(statistics.mean(len(r) for r in reads))
    print(f"generated {len(reads)} IGH reads (avg {avg_len} nt); devices={devices}")

    # ---- prediction throughput (desktop arch) ----
    pred_rows = []
    for device in devices:
        loaded = LoadedModel(make_model("desktop", device), rs_full, ["HUMAN_IGH_OGRDB"],
                             "IGH", None, device)
        for full, vr in [(True, "learned"), (False, "learned"), (False, "parasail")]:
            rps = time_predict(loaded, reads, full_alignment=full, v_reader=vr)
            mem = peak_mem(device, lambda: predict(loaded, reads[:64], full_alignment=full,
                                                   v_reader=vr))
            pred_rows.append((device, full, vr, rps, mem))
            print(f"  predict {device:4s} full={int(full)} vr={vr:8s} -> {rps:6.1f} reads/s, "
                  f"{mem:.2f} GB")
        del loaded
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ---- training step time ----
    train_rows = []
    for preset in ("desktop", "standard"):
        for device in devices:
            if preset == "standard" and device == "cpu":
                continue                                          # impractically slow; omit
            s_step, mem = time_train(preset, device)
            train_rows.append((preset, device, s_step, mem))
            print(f"  train {preset:8s} {device:4s} -> {s_step:.2f}s/step, {mem:.2f} GB")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    write_md(pred_rows, train_rows, avg_len, len(reads), len(rs_full.gene("V").names))


def _cpu_name():
    try:                                                  # platform.processor() is empty on Linux
        for line in open("/proc/cpuinfo"):
            if line.startswith("model name"):
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def write_md(pred_rows, train_rows, avg_len, n, n_v):
    cpu = _cpu_name()
    try:
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        gpu = None
    fa = {True: "yes", False: "no"}
    lines = [
        "# Performance",
        "",
        "Indicative resource numbers for AlignAIR on one machine. Speed depends on the model "
        "architecture, read length, and reference size (not on the trained weights), so these "
        "are measured with **untrained** representative models — use them for planning, not as "
        "accuracy figures (see [Benchmarks](benchmarks.md) for accuracy).",
        "",
        f"Measured on: **{cpu}**" + (f" + **{gpu}**" if gpu else "") + ".  ",
        f"Workload: {n} human-IGH reads (avg {avg_len} nt), full {n_v}-allele V reference, batch 64.  ",
        "Reproduce: `PYTHONPATH=src .venv/bin/python scripts/perf_table.py`.",
        "",
        "## Prediction throughput (desktop model)",
        "",
        "| device | gapped alignment | V reader | reads/s | peak mem |",
        "| --- | --- | --- | --- | --- |",
    ]
    for device, full, vr, rps, mem in pred_rows:
        lines.append(f"| {device.upper()} | {fa[full]} | {vr} | {rps:.0f} | {mem:.2f} GB |")
    lines += [
        "",
        "`--no-full-alignment` skips the parasail gapped alignment (exact cigars / "
        "germline_alignment / identity) for the faster coordinate approximation; "
        "`--v-reader parasail` swaps the learned V reader for the classical one. The two "
        "together are the fastest configuration.",
        "",
        "## Training step time",
        "",
        "| preset | device | s / step | est. wall (preset steps) | peak mem |",
        "| --- | --- | --- | --- | --- |",
    ]
    preset_steps = {"desktop": 3000, "standard": 8000}
    for preset, device, s_step, mem in train_rows:
        eta = s_step * preset_steps[preset]
        h, rem = divmod(int(eta), 3600); mnt = rem // 60
        wall = f"{h}h{mnt:02d}m" if h else f"{mnt}m{int(eta) % 60:02d}s"
        lines.append(f"| {preset} | {device.upper()} | {s_step:.2f} | ~{wall} | {mem:.2f} GB |")
    lines += [
        "",
        "For the small `desktop` model, each step is **data-generation bound** (the GenAIRR "
        "simulation + target building dominate), so CPU and GPU step times are close; the GPU "
        "pulls ahead on the larger `standard` model and bigger batches. The `standard` preset on "
        "CPU is impractically slow and is omitted. Add calibration time (a few minutes) unless "
        "`--no-calibrate`. Estimate your own run first with `alignair train --plan`.",
        "",
    ]
    import os
    path = os.path.join(os.path.dirname(__file__), "..", "docs", "performance.md")
    with open(os.path.abspath(path), "w") as f:
        f.write("\n".join(lines))
    print(f"\nwrote {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
