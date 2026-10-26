"""Safe AA `.alignair` loader: integrity verification + calibration<->model cryptographic binding.

`load_aa_model` resolves a model id/path to a v2 AA bundle (as written by
`alignair.aa.product.bundle.save_aa_bundle`) and loads it with ZERO pickle. Every section is read via
`model_file.container.read_section`, which self-verifies that section's own stored checksum; on top of
that this module INDEPENDENTLY RECOMPUTES the AA integrity manifest (the `reference_provenance.*` /
`integrity.*` hashes the header claims) from the freshly-read section bytes, so a header edited
out-of-band from its sections (e.g. a swapped section whose own index hashes were patched up, but a
stale duplicate hash elsewhere in the header was left behind) is caught too, not just a raw bit-flip.

On top of integrity, the embedded **calibration is cryptographically bound to the exact model it was
calibrated against**: `calibration["provenance"]["model_sha256"]` must equal the header's
`reference_provenance.source_model_sha256` (the sha256 of the *source* `.pt` checkpoint the weights were
loaded from at build time — see `bundle.py`'s docstring for why that is the binding anchor rather than
`integrity.weights_sha256`, which hashes the re-serialized safetensors bytes: same weights, two different
hash domains), the calibration's `class_map_db_hash` must match the model's, and every gene's label count
must agree across the header, the loaded class maps, and the model's own classification-head widths. A
calibration that does not carry a complete method/coverage record is rejected outright. Every check here
RAISES (never warns) — a caller holding an `AAPredictor` has a verified guarantee, not a best-effort one.

Everything is driven off the bundle's own sections; nothing here touches GenAIRR or the network, so a
verified bundle loads to an identical `AAPredictor` anywhere, offline, forever.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

from ...errors import ModelError

from ...aligner import resolve_device
from ...model_file import container, serialize
from ..conformal import FrozenConformal
from ..equivalence import EquivalenceClassMap
from ..model import AAAlignAIR
from ..tokenizer import AAProteinTokenizer
from .bundle import canonical_json
from .predictor import _predict

# Hard-coded model-id -> local bundle path map. Task 8 replaces this with real registry resolution
# (catalog ids resolved through `alignair.registry`, same as the nucleotide `Aligner.from_pretrained`);
# for now a bare filesystem path is always used as-is, and these two aliases cover the Task-2 v2 bundle.
_REPO_ROOT = Path(__file__).resolve().parents[4]
_V2_BUNDLE_PATH = _REPO_ROOT / "experiments" / "aa" / "product" / "alignair-aa-igh-human-2.0.0.alignair"
_MODEL_MAP = {
    "alignair-aa-igh-human@2.0.0": _V2_BUNDLE_PATH,
    "alignair-aa-igh-human": _V2_BUNDLE_PATH,
}


@dataclass
class AAPredictor:
    """A fully-verified, ready-to-run AA model plus everything needed to turn its raw outputs into
    calibrated protein-class -> nucleotide-allele calls.

    ``conformal`` shape: ``{gene: {"pooled": FrozenConformal, "mondrian": {group: FrozenConformal}}}``
    for every gene present in the embedded calibration ("V", "J", and "D" iff ``cfg.has_d``). Callers
    pick ``conformal[gene]["pooled"]`` for the marginal (pooled) coverage guarantee, or
    ``conformal[gene]["mondrian"][group]`` for the group-conditioned one (falling back to pooled
    themselves if the caller's group was not one of the fitted Mondrian groups — this holder does not
    pick a default group, that policy belongs to the caller / Task 7). All `FrozenConformal` instances
    for a gene share that gene's frozen `lambda`/`k_reg` (APS unless the calibration says otherwise).
    """
    model: AAAlignAIR
    cfg: object
    tokenizer: AAProteinTokenizer
    class_maps: dict            # {"V": EquivalenceClassMap, "J": EquivalenceClassMap}
    d_allele_order: list
    asc_multi: dict
    conformal: dict             # {gene: {"pooled": FrozenConformal, "mondrian": {group: FrozenConformal}}}
    card: dict                  # the verified header (== container.read_header, sans "_sections_base")
    device: str

    def predict(self, sequences, *, coverage_group=None, on_invalid="error") -> list:
        """§9 ``predict_aa`` core (Task 5 + Task 7): validated V/D/J calls + §5 coordinates +
        inter_vj_aa + §7 calibrated conformal sets, one `AAPrediction`/`ErrorResult` per input record
        in original order — see `alignair.aa.product.predictor._predict` for the full contract."""
        return _predict(self, sequences, coverage_group=coverage_group, on_invalid=on_invalid)


def _resolve_path(model) -> str:
    if os.path.exists(str(model)):
        return str(model)
    mapped = _MODEL_MAP.get(str(model))
    if mapped is not None:
        return str(mapped)
    raise ModelError(f"unknown AA model id or path: {model!r}", context={"model_id": model})


def _require(cond: bool, message: str) -> None:
    if not cond:
        raise ModelError(message)


def _integrity_manifest(header: dict, *, class_maps_bytes: bytes, asc_bytes: bytes,
                        tokenizer_bytes: bytes, weights_bytes: bytes, calibration_bytes: bytes,
                        config_bytes: bytes) -> None:
    """Recompute every AA integrity-manifest hash from the freshly-read section bytes and assert
    equality with what the header claims (`container.read_section` already checked each section against
    its OWN stored checksum before this ever runs; this additionally catches a header edited
    out-of-band from its sections)."""
    rp = header.get("reference_provenance") or {}
    integ = header.get("integrity") or {}

    class_maps_obj = json.loads(class_maps_bytes.decode("utf-8"))
    # rebuilt from the exact three keys `bundle.py` hashed, per canonical_json (sorted keys, so this is
    # order-independent of how the section happened to serialize them).
    recomputed = container.sha256_hex(canonical_json({
        "V": class_maps_obj.get("V"), "J": class_maps_obj.get("J"),
        "D_alleles": class_maps_obj.get("D_alleles"),
    }))
    _require(recomputed == rp.get("class_map_sha256"), "integrity: class_map_sha256")
    _require(container.sha256_hex(asc_bytes) == rp.get("asc_table_sha256"), "integrity: asc_table_sha256")
    _require(container.sha256_hex(tokenizer_bytes) == rp.get("tokenizer_sha256"),
             "integrity: tokenizer_sha256")
    _require(container.sha256_hex(weights_bytes) == integ.get("weights_sha256"),
             "integrity: weights_sha256")
    _require(container.sha256_hex(calibration_bytes) == integ.get("calibration_sha256"),
             "integrity: calibration_sha256")
    _require(container.sha256_hex(config_bytes) == integ.get("config_sha256"),
             "integrity: config_sha256")


def _rebuild_tokenizer(tokenizer_section: dict) -> AAProteinTokenizer:
    """Reconstruct the tokenizer's char->id table FROM the bundle's own tokenizer section (not from the
    module's current default), so a bundle stays self-describing even if the live alphabet ever drifts."""
    token_dict = {ch: i + 1 for i, ch in enumerate(tokenizer_section["alphabet"])}
    token_dict.update({k: v for k, v in tokenizer_section["special"].items() if k != "PAD"})
    return AAProteinTokenizer(token_dict)


def _check_calibration_complete(calibration: dict, genes: tuple) -> None:
    method = calibration.get("method")
    _require(isinstance(method, dict), "calibration incomplete: method")
    for field in ("name", "alpha", "lambda", "k_reg"):
        _require(method.get(field) is not None, f"calibration incomplete: method.{field}")
    _require(calibration.get("n_cal") is not None, "calibration incomplete: n_cal")
    _require(calibration.get("n_test") is not None, "calibration incomplete: n_test")
    prov = calibration.get("provenance")
    _require(isinstance(prov, dict), "calibration incomplete: provenance")
    _require(prov.get("supported_distribution") is not None,
             "calibration incomplete: provenance.supported_distribution")
    genes_block = calibration.get("genes")
    _require(isinstance(genes_block, dict), "calibration incomplete: genes")
    for gene in genes:
        g = genes_block.get(gene)
        _require(isinstance(g, dict), f"calibration incomplete: genes.{gene}")
        frozen = g.get("calibration_frozen")
        _require(isinstance(frozen, dict), f"calibration incomplete: genes.{gene}.calibration_frozen")
        _require(frozen.get("q_pooled") is not None,
                 f"calibration incomplete: genes.{gene}.calibration_frozen.q_pooled")
        q_mondrian = frozen.get("q_mondrian")
        _require(isinstance(q_mondrian, dict) and len(q_mondrian) > 0,
                 f"calibration incomplete: genes.{gene}.calibration_frozen.q_mondrian")


def _require_nonempty_hex(value, message: str) -> None:
    """Presence + type gate for a binding-anchor hash field, so that two ABSENT fields (both `None`)
    never satisfy a later `==` comparison -- a bare equality check alone is vacuously true when both
    sides are missing, which is exactly how a header/calibration can have its binding fields deleted
    outright and still "match"."""
    _require(isinstance(value, str) and len(value) > 0, message)


def _check_binding(header: dict, calibration: dict, *, class_maps: dict, d_allele_order: list,
                   cfg, model: AAAlignAIR) -> tuple:
    """Calibration<->model cryptographic binding + label-count agreement. Returns the tuple of gene
    names present (``("V","J")`` or ``("V","J","D")``) for the caller to build the conformal holder
    from, so the same "which genes exist" logic is not duplicated."""
    rp = header.get("reference_provenance") or {}
    prov = calibration.get("provenance") or {}

    cal_model_sha = prov.get("model_sha256")
    hdr_model_sha = rp.get("source_model_sha256")
    _require_nonempty_hex(cal_model_sha,
             "calibration<->model binding: provenance.model_sha256 missing")
    _require_nonempty_hex(hdr_model_sha,
             "calibration<->model binding: reference_provenance.source_model_sha256 missing")
    _require(cal_model_sha == hdr_model_sha,
             "calibration<->model binding: provenance.model_sha256 does not match this model's "
             "reference_provenance.source_model_sha256 (calibration was fit against a different model)")

    # Anchor on the REAL, checksum-verified class maps (`class_maps[gene].db_hash`, loaded via
    # `EquivalenceClassMap.from_json` from the `class_maps` section whose bytes `_integrity_manifest`
    # already verified) -- NOT on the header's `reference_provenance.class_map_db_hash`, which is just
    # another editable header field. Comparing calibration-claim against header-claim only proves the
    # two header/section duplicates agree with EACH OTHER, not that either matches the actual reference
    # data the model was built against.
    cal_db_hash = prov.get("class_map_db_hash") or {}
    header_db_hash = rp.get("class_map_db_hash") or {}
    for gene in ("V", "J"):
        real_hash = class_maps[gene].db_hash
        cal_hash = cal_db_hash.get(gene)
        _require_nonempty_hex(real_hash,
                 f"calibration<->model binding: class_maps.{gene}.db_hash missing")
        _require_nonempty_hex(cal_hash,
                 f"calibration<->model binding: provenance.class_map_db_hash.{gene} missing")
        _require(cal_hash == real_hash,
                 f"calibration<->model binding: provenance.class_map_db_hash.{gene} does not match "
                 f"the verified class_maps.{gene}.db_hash (calibration was fit against a different "
                 f"reference set)")
        # advisory: the header's own duplicate claim should also agree with the real data (catches a
        # header edited independently of both the calibration and the class_maps section itself).
        _require_nonempty_hex(header_db_hash.get(gene),
                 f"calibration<->model binding: reference_provenance.class_map_db_hash.{gene} missing")
        _require(header_db_hash.get(gene) == real_hash,
                 f"calibration<->model binding: reference_provenance.class_map_db_hash.{gene} does not "
                 f"match the verified class_maps.{gene}.db_hash")

    header_counts = header.get("class_counts") or {}
    head_counts = {name: branch.cls_head.out_features for name, branch in model.branches.items()}
    _require(header_counts.get("v") == class_maps["V"].n_classes() == head_counts.get("v"),
             "calibration<->model binding: V label-count mismatch across header/class_map/model head")
    _require(header_counts.get("j") == class_maps["J"].n_classes() == head_counts.get("j"),
             "calibration<->model binding: J label-count mismatch across header/class_map/model head")
    genes = ("V", "J")
    if cfg.has_d:
        _require(header_counts.get("d") == len(d_allele_order) == head_counts.get("d"),
                 "calibration<->model binding: D label-count mismatch across header/class_map/model head")
        genes = ("V", "J", "D")

    _check_calibration_complete(calibration, genes)
    return genes


def _conformal_holder(calibration: dict, genes: tuple) -> dict:
    """``{gene: {"pooled": FrozenConformal, "mondrian": {group: FrozenConformal}}}`` — see
    `AAPredictor.conformal` for the contract this shape satisfies."""
    method = calibration["method"]
    lam, k_reg = method["lambda"], method["k_reg"]
    out = {}
    for gene in genes:
        frozen = calibration["genes"][gene]["calibration_frozen"]
        out[gene] = {
            "pooled": FrozenConformal(frozen["q_pooled"], lam=lam, k_reg=k_reg),
            "mondrian": {group: FrozenConformal(q, lam=lam, k_reg=k_reg)
                        for group, q in frozen["q_mondrian"].items()},
        }
    return out


def _load_legacy_pickle(path: str, *, device: str) -> AAPredictor:
    """Dev-only convenience load of a raw ``strategic._save``-style checkpoint (``{"model", "cfg",
    "step"}``, `torch.save` pickle). It carries no class maps / ASC table / calibration, so those come
    back empty — there is no calibration to bind, and no conformal sets can be built. Only reachable
    with an explicit ``trust_pickle=True``; never used by the verified v2-bundle path above."""
    import dataclasses

    import torch

    from ...core.config import AlignAIRConfig

    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg_raw = ck["cfg"]
    cfg = cfg_raw if not isinstance(cfg_raw, dict) else AlignAIRConfig(
        **{f.name: cfg_raw[f.name] for f in dataclasses.fields(AlignAIRConfig) if f.name in cfg_raw})
    dev = resolve_device(device)
    model = AAAlignAIR(cfg)
    model.load_state_dict(ck["model"], strict=True)
    model.to(dev).eval()
    return AAPredictor(model=model, cfg=cfg, tokenizer=AAProteinTokenizer(), class_maps={},
                       d_allele_order=[], asc_multi={}, conformal={},
                       card={"artifact_type": "legacy-pt-dev-load", "trust_pickle": True, "path": path},
                       device=dev)


def load_aa_model(model, *, device: str = "auto", trust_pickle: bool = False) -> AAPredictor:
    """Resolve ``model`` (a filesystem path, or a hard-coded catalog id — see `_MODEL_MAP`) and load it
    into a verified `AAPredictor`. Raises (never warns) on any integrity or calibration<->model binding
    failure. A legacy pickle ``.pt`` checkpoint is refused unless ``trust_pickle=True`` (a dev-only,
    integrity-unverified convenience path — see `_load_legacy_pickle`)."""
    path = _resolve_path(model)

    if not container.is_alignair_file(path):
        if not trust_pickle:
            raise ModelError(
                f"{path!r} is not a v2 .alignair bundle (looks like a legacy pickle checkpoint); pass "
                "trust_pickle=True to load it anyway (dev-only, integrity-unverified path).")
        return _load_legacy_pickle(path, device=device)

    header = container.read_header(path)
    header.pop("_sections_base", None)

    weights_bytes = container.read_section(path, "weights")           # each read_section call already
    config_bytes = container.read_section(path, "config")             # self-verifies that section's own
    tokenizer_bytes = container.read_section(path, "tokenizer")       # stored (compressed + payload)
    class_maps_bytes = container.read_section(path, "class_maps")     # checksum -- corrupt/truncated
    asc_bytes = container.read_section(path, "asc_imgt")               # bytes raise right here.
    calibration_bytes = container.read_section(path, "calibration")

    _integrity_manifest(header, class_maps_bytes=class_maps_bytes, asc_bytes=asc_bytes,
                        tokenizer_bytes=tokenizer_bytes, weights_bytes=weights_bytes,
                        calibration_bytes=calibration_bytes, config_bytes=config_bytes)

    cfg = serialize.config_from_bytes(config_bytes)
    class_maps_obj = json.loads(class_maps_bytes.decode("utf-8"))
    class_maps = {"V": EquivalenceClassMap.from_json(class_maps_obj["V"]),
                 "J": EquivalenceClassMap.from_json(class_maps_obj["J"])}
    d_allele_order = list(class_maps_obj["D_alleles"])
    asc_multi = json.loads(asc_bytes.decode("utf-8"))
    calibration = json.loads(calibration_bytes.decode("utf-8"))
    tokenizer = _rebuild_tokenizer(json.loads(tokenizer_bytes.decode("utf-8")))

    dev = resolve_device(device)
    model_obj = AAAlignAIR(cfg)
    model_obj.load_state_dict(serialize.state_dict_from_bytes(weights_bytes), strict=True)
    model_obj.to(dev).eval()

    genes = _check_binding(header, calibration, class_maps=class_maps, d_allele_order=d_allele_order,
                           cfg=cfg, model=model_obj)
    conformal = _conformal_holder(calibration, genes)

    return AAPredictor(model=model_obj, cfg=cfg, tokenizer=tokenizer, class_maps=class_maps,
                       d_allele_order=d_allele_order, asc_multi=asc_multi, conformal=conformal,
                       card=header, device=dev)
