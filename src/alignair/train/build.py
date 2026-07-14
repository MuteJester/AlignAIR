"""Custom-reference training support (AIRR-review #6): build a GenAIRR DataConfig from user V/D/J
FASTAs, produce a pre-training plan (validate-only), and export a safe **pickle-free** model bundle
(model + model card + reference manifest + validation report) after training.
"""
from __future__ import annotations

import json
import os
from typing import Optional


def build_dataconfigs(*, dataconfig: Optional[list] = None, v_fasta: Optional[str] = None,
                      d_fasta: Optional[str] = None, j_fasta: Optional[str] = None,
                      chain_type: Optional[str] = None):
    """Resolve the training reference to a list of GenAIRR DataConfigs, from built-in names OR a custom
    V/D/J FASTA set. Returns ``(dataconfigs, cartridge_report_or_None)``."""
    if dataconfig:
        import GenAIRR.data as gd
        try:
            return [getattr(gd, d) for d in dataconfig], None
        except AttributeError as e:
            raise ValueError(f"unknown built-in dataconfig: {e}. Run `alignair reference list`.") from e
    if not (v_fasta and j_fasta and chain_type):
        raise ValueError("custom training needs --v-fasta, --j-fasta and --chain-type "
                         "(plus --d-fasta for heavy/D-bearing loci)")
    for label, path in (("--v-fasta", v_fasta), ("--j-fasta", j_fasta), ("--d-fasta", d_fasta)):
        if path and not os.path.exists(path):
            raise ValueError(f"{label} file not found: {path}")
    from GenAIRR.cartridge_builder import ReferenceCartridgeBuilder
    builder = ReferenceCartridgeBuilder.from_fasta(v_fasta=v_fasta, j_fasta=j_fasta, d_fasta=d_fasta,
                                                   chain_type=chain_type)
    return [builder.build()], builder.report()


def training_plan(dcs, *, steps: int, batch_size: int) -> dict:
    """A pre-flight plan (no training): reference composition, model parameter count, and the resolved
    schedule — so a user can sanity-check a custom reference / preset before committing GPU time."""
    from ..core import AlignAIR
    from ..core.config import AlignAIRConfig
    from ..reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(*dcs)
    cfg = AlignAIRConfig.from_dataconfigs(*dcs)
    params = sum(p.numel() for p in AlignAIR(cfg).parameters())
    return {
        "loci": list(ref.locus_names()),
        "alleles": {G: len(ref.gene(G)) for G in ("V", "D", "J") if G in ref.genes},
        "has_d": ref.has_d, "max_seq_length": cfg.max_seq_length,
        "model_parameters": params, "steps": steps, "batch_size": batch_size,
    }


def _validation_report(checkpoint_path: str, dcs, *, n_batches: int = 2, batch_size: int = 32,
                       seed: int = 424242) -> dict:
    """Evaluate the trained model on a fixed held-out stream and return per-task metrics."""
    import itertools

    import torch

    from ..api import load_model
    from ..train.gym import Curriculum, build_experiment
    from ..train.trainer import build_batch, eval_metrics
    model, ref = load_model(checkpoint_path, device="cpu")
    agg: dict = {}
    for i, dc in enumerate(dcs):
        exp = build_experiment(dc, dict(Curriculum().params(0.3)), allow_curatable=True)
        recs = [r for r in itertools.islice(exp.stream_records(n=None, seed=seed + i), batch_size * n_batches * 3)
                if all(r.get(f"{g}_sequence_start") is not None for g in (("v", "d", "j") if ref.has_d else ("v", "j")))]
        for b in range(0, len(recs) - batch_size, batch_size):
            batch_in, targets = build_batch(recs[b:b + batch_size], ref, model.cfg, device="cpu")
            with torch.no_grad():
                for k, v in eval_metrics(model(batch_in), targets, model.cfg).items():
                    agg.setdefault(k, []).append(v)
    return {k: round(sum(vs) / len(vs), 4) for k, vs in agg.items() if vs}


def export_bundle(checkpoint_path: str, dcs, bundle_dir: str, *, training: dict,
                  model_id: Optional[str] = None, description: str = "", validate: bool = True) -> str:
    """Export a **pickle-free**, distributable model bundle from a (resumable) training checkpoint:
    ``model.alignair`` (no trusted pickle), ``model_card.md``, ``reference_manifest.json`` and (unless
    ``validate=False``) ``validation_report.json``. Returns the model path."""
    from .. import model_file as mf
    from ..api import load_model
    os.makedirs(bundle_dir, exist_ok=True)
    model, ref = load_model(checkpoint_path, device="cpu")     # safe inference load (no pickle needed)
    model_path = os.path.join(bundle_dir, "model.alignair")

    report = _validation_report(checkpoint_path, dcs) if validate else {}
    card = {"training": training, "validation": report}
    mf.save_model(model_path, model, dataconfigs=dcs, training=training,
                  include_trusted_pickle=False, model_id=model_id, description=description, card=card)

    manifest = {"loci": list(ref.locus_names()),
                "genes": {G: {"n": len(ref.gene(G)), "names": list(ref.gene(G).names)}
                          for G in ("V", "D", "J") if G in ref.genes}}
    with open(os.path.join(bundle_dir, "reference_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    if validate:
        with open(os.path.join(bundle_dir, "validation_report.json"), "w") as f:
            json.dump(report, f, indent=2)
    _write_card(os.path.join(bundle_dir, "model_card.md"), training, manifest, report)
    return model_path


def _write_card(path: str, training: dict, manifest: dict, report: dict) -> None:
    lines = ["# AlignAIR model card", "",
             f"- loci: {', '.join(manifest['loci']) or 'n/a'}",
             f"- alleles: " + ", ".join(f"{G}={d['n']}" for G, d in manifest["genes"].items()),
             f"- training steps: {training.get('steps')}", ""]
    if report:
        lines += ["## Validation (fixed held-out stream)", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(report.items())]
    lines += ["", "This is a fixed-reference classifier: the embedded reference is the callable set.",
              "Adding alleles / species requires training a new model. See docs/architecture/model_contract.md."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
