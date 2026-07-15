"""Custom-reference training support (AIRR-review #6): build a GenAIRR DataConfig from user V/D/J
FASTAs, produce a pre-training plan (validate-only), and export a safe **pickle-free** model bundle
(model + model card + reference manifest + validation report) after training.
"""
from __future__ import annotations

import json
import os
from typing import Optional


# chain types that carry a D gene (require --d-fasta); the rest must NOT be given one
_HAS_D_CHAINS = {"BCR_HEAVY", "TCR_BETA", "TCR_DELTA"}


def build_dataconfigs(*, dataconfig: Optional[list] = None, v_fasta: Optional[str] = None,
                      d_fasta: Optional[str] = None, j_fasta: Optional[str] = None,
                      chain_type: Optional[str] = None):
    """Resolve the training reference to a list of GenAIRR DataConfigs, from built-in names OR a custom
    V/D/J FASTA set (exactly one mode). Returns ``(dataconfigs, cartridge_report_or_None)``."""
    custom = any((v_fasta, d_fasta, j_fasta, chain_type))
    if dataconfig and custom:
        raise ValueError("choose ONE reference mode: --dataconfig (built-in) OR a custom "
                         "--v-fasta/--j-fasta/--chain-type — not both")
    if dataconfig:
        import GenAIRR.data as gd
        try:
            return [getattr(gd, d) for d in dataconfig], None
        except AttributeError as e:
            raise ValueError(f"unknown built-in dataconfig: {e}. Run `alignair reference list`.") from e
    if not (v_fasta and j_fasta and chain_type):
        raise ValueError("custom training needs --v-fasta, --j-fasta and --chain-type "
                         "(plus --d-fasta for heavy / D-bearing loci)")
    ct = str(chain_type).upper()
    has_d = ct in _HAS_D_CHAINS
    if has_d and not d_fasta:
        raise ValueError(f"chain type {ct} carries a D gene — provide --d-fasta")
    if not has_d and d_fasta:
        raise ValueError(f"chain type {ct} has no D gene — do not pass --d-fasta")
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
        for b in range(0, len(recs) - batch_size + 1, batch_size):    # +1: include the last full batch
            batch_in, targets = build_batch(recs[b:b + batch_size], ref, model.cfg, device="cpu")
            with torch.no_grad():
                for k, v in eval_metrics(model(batch_in), targets, model.cfg).items():
                    agg.setdefault(k, []).append(v)
    return {k: round(sum(vs) / len(vs), 4) for k, vs in agg.items() if vs}


def _sha256(path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _reference_manifest(ref, sources, report, model_path) -> dict:
    """A standalone reference fingerprint: allele order hash, per-gene anchor coverage, source FASTA
    hashes, tool versions, and the exported model artifact hash — enough to identify the exact reference
    without opening the model container."""
    import GenAIRR

    from .. import __version__
    from ..model_file import serialize
    genes = {}
    for G in ("V", "D", "J"):
        if G not in ref.genes:
            continue
        gr = ref.gene(G)
        anchored = sum(1 for n in gr.names if gr.anchors and n in gr.anchors)
        genes[G] = {"n": len(gr), "anchored": anchored, "names": list(gr.names)}
    manifest = {
        "loci": [{"locus": l.locus, "chain_type": l.chain_type, "has_d": l.has_d} for l in ref.loci],
        "genes": genes,
        # canonical fingerprint of the reconstructed reference itself — identifies the exact allele set
        # independently of source-file layout (matches the container's embedded-reference check).
        "reference_fasta_sha256": serialize.reference_fasta_sha256(ref),
        "allele_order_sha256": serialize.allele_order_sha256(ref),
        "versions": {"alignair": __version__, "genairr": getattr(GenAIRR, "__version__", "unknown")},
        "model_artifact_sha256": _sha256(model_path),
    }
    if sources:
        manifest["sources"] = {k: {"path": v, "sha256": _sha256(v)} for k, v in sources.items() if v}
    if report is not None:
        manifest["reference_build"] = {"warnings": list(report.warnings), "rejected": len(report.rejected)}
    return manifest


def export_bundle(checkpoint_path: str, dcs, bundle_dir: str, *, training: dict,
                  model_id: Optional[str] = None, description: str = "", validate: bool = True,
                  sources: Optional[dict] = None, report=None, overwrite: bool = False) -> str:
    """Export a **pickle-free**, distributable model bundle from a (resumable) training checkpoint:
    ``model.alignair`` (no trusted pickle) + ``model_card.md`` + ``reference_manifest.json`` + (unless
    ``validate=False``) ``validation_report.json``. Returns the model path.

    Interruption-safe (AIRR-review): built in a sibling temp directory and renamed onto ``bundle_dir``
    only after every file succeeds and the model reloads. **Fails closed** — if ``bundle_dir`` already
    exists it is NOT overwritten unless ``overwrite=True`` (a fresh output dir is the normal path, so a
    published bundle is never silently destroyed). With ``overwrite=True`` the existing bundle is moved
    to a sibling ``.backup.<pid>`` first and **rolled back if the swap raises** — but this is rollback-safe
    on a caught failure, NOT crash-atomic: a hard process kill in the small window between moving the old
    bundle aside and installing the new one can leave only the ``.backup`` copy (recoverable by hand)."""
    import shutil
    import tempfile

    from .. import model_file as mf
    from ..api import load_model
    bundle_dir = os.path.abspath(bundle_dir)
    parent = os.path.dirname(bundle_dir) or "."
    os.makedirs(parent, exist_ok=True)
    if os.path.exists(bundle_dir) and not overwrite:             # fail fast, before any GPU/validation
        raise FileExistsError(
            f"bundle directory already exists: {bundle_dir}. Refusing to overwrite a possibly-published "
            f"bundle — choose a new output directory or pass overwrite=True (--overwrite).")
    # build in a secure sibling temp dir on the SAME filesystem (so the final rename is atomic)
    tmp = tempfile.mkdtemp(dir=parent, prefix=os.path.basename(bundle_dir) + ".building.")
    try:
        model, ref = load_model(checkpoint_path, device="cpu")   # safe inference load (no pickle needed)
        metrics = _validation_report(checkpoint_path, dcs) if validate else {}
        if validate and not metrics:                             # empty report is NOT success
            raise ValueError("validation produced no metrics (too few held-out records for a full "
                             "batch); lower the validation batch size or disable validation")
        model_path = os.path.join(tmp, "model.alignair")
        mf.save_model(model_path, model, dataconfigs=dcs, training=training,
                      include_trusted_pickle=False, model_id=model_id, description=description,
                      card={"training": training, "validation": metrics})
        load_model(model_path, device="cpu")                     # verify the export reloads

        manifest = _reference_manifest(ref, sources, report, model_path)
        with open(os.path.join(tmp, "reference_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if validate:
            with open(os.path.join(tmp, "validation_report.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        _write_card(os.path.join(tmp, "model_card.md"), training, manifest, metrics)

        # commit: move any existing bundle aside (overwrite=True), swap in the new one, restore on failure
        backup = None
        if os.path.exists(bundle_dir):
            backup = f"{bundle_dir}.backup.{os.getpid()}"
            shutil.rmtree(backup, ignore_errors=True)
            os.replace(bundle_dir, backup)                       # fresh dst name -> clean rename
        try:
            os.replace(tmp, bundle_dir)                          # destination now absent -> atomic
        except BaseException:
            if backup is not None and not os.path.exists(bundle_dir):
                os.replace(backup, bundle_dir)                   # roll back to the previous bundle
            raise
        if backup is not None:
            shutil.rmtree(backup, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return os.path.join(bundle_dir, "model.alignair")


def _write_card(path: str, training: dict, manifest: dict, report: dict) -> None:
    loci = ", ".join(l["locus"] for l in manifest.get("loci", [])) or "n/a"
    lines = ["# AlignAIR model card", "",
             f"- loci: {loci}",
             "- alleles: " + ", ".join(f"{G}={d['n']}" for G, d in manifest["genes"].items()),
             f"- training steps: {training.get('steps')}", ""]
    if report:
        lines += ["## Validation (fixed held-out stream)", ""]
        lines += [f"- {k}: {v}" for k, v in sorted(report.items())]
    lines += ["", "This is a fixed-reference classifier: the embedded reference is the callable set.",
              "Adding alleles / species requires training a new model. See docs/model_contract.md."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
