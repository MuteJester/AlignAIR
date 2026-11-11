"""Custom-reference training support: build a GenAIRR DataConfig from user V/D/J
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
        from ..errors import ConfigurationError
        from ..genairr import resolve_dataconfigs
        try:
            return resolve_dataconfigs(dataconfig), None
        except ConfigurationError as e:
            raise ValueError(f"unknown built-in dataconfig: {e.context.get('dataconfig')}. "
                             "Run `alignair reference list`.") from e
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
    from ..core import prepare_model
    from ..reference.reference_set import ReferenceSet
    ref = ReferenceSet.from_dataconfigs(*dcs)
    setup = prepare_model(ref)
    cfg = setup.config
    params = sum(p.numel() for p in setup.build_model().parameters())
    return {
        "loci": list(ref.locus_names()),
        "alleles": {G: len(ref.gene(G)) for G in ("V", "D", "J") if G in ref.genes},
        "has_d": ref.has_d, "max_seq_length": cfg.max_seq_length,
        "model_parameters": params, "steps": steps, "batch_size": batch_size,
    }


def _checkpoint_training(checkpoint_path: str) -> dict:
    """The training provenance recorded IN the checkpoint (actual steps, lr, curriculum progresses,
    heavy_shm, seed, and ``train_args`` incl. the effective per-locus SHM caps).

    The checkpoint is the source of truth: a caller that hand-rebuilds a training dict silently drops
    every field it does not know about — which is how the distributable bundle used to lose
    ``effective_mutation_caps``. Returns ``{}`` for a legacy/unreadable artifact so export still works."""
    try:
        from .. import model_file as mf
        return dict(mf.read_metadata(checkpoint_path).get("training") or {})
    except Exception:                                  # noqa: BLE001 - provenance must never block export
        return {}


def required_segments(dataconfig) -> tuple:
    """The segments a validation record must carry coordinates for, for THIS locus.

    Derived per dataconfig, never from the union reference: in a mixed D / non-D run (e.g. TRA+TRB) the
    union's ``has_d`` is True, which would drop EVERY TRA record for lacking D coordinates and silently
    reduce the validation report to the D-bearing locus alone."""
    return ("v", "d", "j") if bool(getattr(dataconfig.metadata, "has_d", False)) else ("v", "j")


def _validation_report(checkpoint_path: str, dcs, *, n_batches: int = 2, batch_size: int = 32,
                       seed: int = 424242) -> dict:
    """Evaluate the trained model on a fixed held-out stream and return per-task metrics."""
    import itertools

    import torch

    from ..api import load_model
    from ..train.gym import Curriculum, build_experiment
    from ..train.trainer import build_batch, eval_metrics
    from ..reference.reference_set import is_tcr
    model, ref = load_model(checkpoint_path, device="cpu")
    agg: dict = {}
    for i, dc in enumerate(dcs):
        params = dict(Curriculum(indel_form="uniform").params(0.3))   # pinned: the eval data never drifts
        if is_tcr(dc):                            # TCR loci do not undergo SHM (GenAIRR forbids mutate())
            params["mutation_rate"] = 0.0
        exp = build_experiment(dc, params, allow_curatable=True)
        segs = required_segments(dc)               # per-locus, not the union reference (see the helper)
        recs = [r for r in itertools.islice(exp.stream_records(n=None, seed=seed + i), batch_size * n_batches * 3)
                if all(r.get(f"{g}_sequence_start") is not None for g in segs)]
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
    from ..genairr import genairr_identity
    genairr = genairr_identity()

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
        "versions": {"alignair": __version__, "genairr": genairr["genairr_version"],
                     "genairr_determinism_version": genairr["genairr_determinism_version"],
                     "genairr_source_sha256": genairr["genairr_source_sha256"]},
        "model_artifact_sha256": _sha256(model_path),
    }
    if sources:
        manifest["sources"] = {k: {"path": v, "sha256": _sha256(v)} for k, v in sources.items() if v}
    if report is not None:
        manifest["reference_build"] = {"warnings": list(report.warnings), "rejected": len(report.rejected)}
    return manifest


def export_bundle(checkpoint_path: str, dcs, bundle_dir: str, *, training: Optional[dict] = None,
                  model_id: Optional[str] = None, model_version: Optional[str] = None, description: str = "",
                  validate: bool = True, sources: Optional[dict] = None, report=None, overwrite: bool = False) -> str:
    """Export a **pickle-free**, distributable model bundle from a (resumable) training checkpoint:
    ``model.alignair`` (no trusted pickle) + ``model_card.md`` + ``reference_manifest.json`` + (unless
    ``validate=False``) ``validation_report.json``. Returns the model path. ``model_id`` and ``model_version`` go
    into the model file's card, where the catalog's ``publish_local`` checks them.

    Training provenance is read FROM ``checkpoint_path`` (:func:`_checkpoint_training`) so the
    distributable bundle carries the same record as the resumable checkpoint — actual steps, curriculum
    progresses, heavy_shm, seed, and the effective per-locus SHM caps. ``training`` is an optional
    override merged on top; callers must not have to restate the schema (a hand-built dict silently
    dropped ``train_args``, and with it the effective caps, from every published bundle).

    Interruption-safe: built in a sibling temp directory and renamed onto ``bundle_dir``
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
    # the checkpoint is the source of truth for provenance; explicit overrides win
    training = {**_checkpoint_training(checkpoint_path), **(training or {})}
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
                      include_trusted_pickle=False, model_id=model_id, model_version=model_version,
                      description=description,
                      card={"training": training, "validation": metrics},
                      label_schema=mf.read_metadata(checkpoint_path).get('model', {}).get('label_schema'))
        load_model(model_path, device="cpu")                     # verify the export reloads

        manifest = _reference_manifest(ref, sources, report, model_path)
        with open(os.path.join(tmp, "reference_manifest.json"), "w") as f:
            json.dump(manifest, f, indent=2)
        if validate:
            with open(os.path.join(tmp, "validation_report.json"), "w") as f:
                json.dump(metrics, f, indent=2)
        _write_card(os.path.join(tmp, "model_card.md"), training, manifest, metrics, dcs=dcs, model_id=model_id,
                    model_version=model_version)

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


def checkpoint_dataconfigs(checkpoint_path: str) -> tuple[list, list[str]]:
    """The GenAIRR dataconfigs a checkpoint was trained on, for ``export_bundle``, and a note per one read from the
    checkpoint's own pickled section. A built-in dataconfig is taken by the name its card records (no unpickling) when
    its schema digest matches the card's; any other (a custom FASTA reference) is read from the checkpoint's
    ``dataconfig/<i>`` section, which unpickles it: export only checkpoints you trust, as for resuming one.
    ConfigurationError when the card names none, or one is neither built-in nor stored."""
    from ..errors import ConfigurationError
    from ..genairr import resolve_dataconfig
    from .. import model_file as mf
    entries = mf.read_metadata(checkpoint_path).get("reference", {}).get("dataconfigs", [])
    if not entries:
        raise ConfigurationError(f"{checkpoint_path} records no dataconfig to export with",
                                 context={"path": checkpoint_path})
    dcs, notes = [], []
    for i, entry in enumerate(entries):
        dc = None
        try:
            dc = resolve_dataconfig(entry.get("name"))
        except ConfigurationError:
            pass
        if dc is not None and entry.get("schema_sha256") not in (None, getattr(dc, "schema_sha256", None)):
            dc = None                                  # a built-in of that name, but not the one trained on
        if dc is None:
            section = entry.get("section")
            if not section:
                raise ConfigurationError(f"{checkpoint_path}: dataconfig {entry.get('name')!r} is not built in and "
                                         "the file stores none (export from the training checkpoint)",
                                         context={"path": checkpoint_path, "dataconfig": entry.get("name")})
            dc = mf.read_dataconfig(checkpoint_path, index=entry.get("index", i))
            notes.append(f"read dataconfig {entry.get('name')!r} from the checkpoint's pickled {section} section")
        dcs.append(dc)
    return dcs, notes


def _reference_line(dc) -> str:
    """``HUMAN_IGH_OGRDB (OGRDB V8, updated 2025-02-01; Human BCR_HEAVY)``: a dataconfig's name and its metadata's
    reference set, date, species and chain type, those it has."""
    from ..genairr import name_of
    md = getattr(dc, "metadata", None)
    value = lambda x: str(getattr(x, "value", x)) if x is not None else None
    source = getattr(md, "reference_set", None)
    updated = getattr(md, "last_updated", None)
    what = " ".join(v for v in (value(getattr(md, "species", None)), value(getattr(md, "chain_type", None))) if v)
    version = (f"{source}" + (f", updated {updated}" if updated else "")) if source else None
    detail = "; ".join(v for v in (version, what or None) if v)
    name = name_of(dc) or "custom reference"
    return f"{name} ({detail})" if detail else name


def _write_card(path: str, training: dict, manifest: dict, report: dict, *, dcs=(), model_id=None,
                model_version=None) -> None:
    """The bundle's ``model_card.md``: the model, its reference by name and version, allele counts, training, intended
    use, usage (genotype and reference swap), the validation metrics (a ``PENDING-V4-NUMBERS`` marker without them) and
    where the licence and citation are."""
    loci = ", ".join(l["locus"] for l in manifest.get("loci", [])) or "n/a"
    title = " ".join(v for v in (model_id, model_version) if v)
    lines = ["# AlignAIR model card", ""]
    if title:
        lines.append(f"- model: {title}")
    lines += [f"- loci: {loci}"]
    lines += [f"- reference: {_reference_line(dc)}" for dc in dcs]
    lines += ["- alleles: " + ", ".join(f"{G}={d['n']}" for G, d in manifest["genes"].items()),
              f"- training steps: {training.get('steps')}"]
    # The SHM cap actually applied to each locus (TCR -> 0). Without this the card misstates the
    # training distribution: a TRA run otherwise reads as an uncapped curriculum with heavy_shm=0.25.
    caps = (training.get("train_args") or {}).get("effective_mutation_caps")
    if caps:
        rendered = ", ".join(f"{k}={'uncapped' if v is None else v}" for k, v in caps.items())
        lines.append(f"- effective SHM cap per locus: {rendered}")
    if training.get("heavy_shm") is not None:
        lines.append(f"- heavy-SHM stream (requested): {training['heavy_shm']}")
    model = model_id or "model.alignair"
    lines += ["", "## Intended use", "",
              f"Aligning single-end {loci} reads (full-length or fragments, any orientation) to AIRR rearrangements. "
              "The embedded reference is the default callable set. The model reads its alleles from the bound "
              "reference, so other alleles of the same genes (a donor's genotype, novel alleles) are called by "
              "binding another reference (alignair predict --reference) without retraining. Not paired-read "
              "assembly; another locus or species needs a model trained for it.", "",
              "## Usage", "", "```bash",
              f"alignair predict --input reads.fasta --out out.tsv --model {model}",
              "# restrict calls to a donor's genotype (YAML/JSON keyed by V, D, J)",
              f"alignair predict --input reads.fasta --out out.tsv --model {model} --genotype donor.yaml",
              "# infer the donor's V genotype from the reads (two passes over the file)",
              f"alignair predict --input reads.fasta --out out.tsv --model {model} --infer-genotype",
              "# call from another reference of these genes, novel alleles included, without retraining",
              f"alignair predict --input reads.fasta --out out.tsv --model {model} --reference germline.fasta",
              "```", "", "## Validation", ""]
    if report:
        lines += ["Fixed held-out stream:", ""] + [f"- {k}: {v}" for k, v in sorted(report.items())]
    else:
        lines += ["<!-- PENDING-V4-NUMBERS -->",
                  "Validation metrics are pending: this bundle was exported without validation."]
    lines += ["", "## License and citation", "",
              "GPL-3.0-or-later: see LICENSE, and cite AlignAIR as CITATION.cff gives, in the AlignAIR repository "
              "(https://github.com/MuteJester/AlignAIR)."]
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
