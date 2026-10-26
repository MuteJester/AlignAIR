import json

import GenAIRR as ga
import pytest
import torch

from alignair.aa.product.asc import asc_to_imgt_multi
from alignair.aa.product.bundle import canonical_json, save_aa_bundle
from alignair.aa.product.loader import AAPredictor, _MODEL_MAP, _resolve_path, load_aa_model
from alignair.aa.strategic import strategic_setup
from alignair.aa.tokenizer import pad_tokenize_aa
from alignair.model_file import container

pytestmark = pytest.mark.component

_REAL_MODEL_ID = "alignair-aa-igh-human@2.0.0"
_FAKE_MODEL_SHA256 = "cafebabe" * 8
_OTHER_MODEL_SHA256 = "deadbeef" * 8


def _gene_block(**overrides):
    block = {
        "calibration_frozen": {
            "groups_fitted": ["full", "short"],
            "pooled_fallback_exercised": False,
            "q_pooled": 0.9,
            "q_mondrian": {"full": 0.91, "short": 0.95},
        },
        "n_test": 10,
    }
    block.update(overrides)
    return block


def _full_calibration(class_maps, d_allele_order, *, model_sha256):
    return {
        "alpha": 0.05,
        "method": {"name": "APS", "alpha": 0.05, "lambda": 0.0, "k_reg": 1},
        "n_cal": 10,
        "n_test": 10,
        "genes": {
            "V": _gene_block(),
            "J": _gene_block(),
            "D": _gene_block(),
        },
        "provenance": {
            "model_sha256": model_sha256,
            "class_map_db_hash": {"V": class_maps["V"].db_hash, "J": class_maps["J"].db_hash},
            "supported_distribution": "SIMULATOR-DOMAIN ONLY: test fixture",
        },
    }


@pytest.fixture(scope="module")
def built():
    """Built once (needs GenAIRR reference data) and reused by every tmp-bundle test in this module —
    only `save_aa_bundle` (cheap: serialize + write) runs per test, not `strategic_setup`."""
    model, cfg, class_maps, meta = strategic_setup("HUMAN_IGH_OGRDB", 192, has_d=True)
    dc = ga.HUMAN_IGH_OGRDB
    asc_multi = asc_to_imgt_multi(dc)
    d_allele_order = [name for name, _ in sorted(meta["d_index"].items(), key=lambda kv: kv[1])]
    return {"model": model, "cfg": cfg, "class_maps": class_maps, "d_allele_order": d_allele_order,
            "asc_multi": asc_multi}


def _write_bundle(built, path, *, model_sha256=_FAKE_MODEL_SHA256, calibration=None,
                  card_overrides=None):
    cal = calibration if calibration is not None else _full_calibration(
        built["class_maps"], built["d_allele_order"], model_sha256=model_sha256)
    card = {"model_id": "alignair-aa-igh-human-test", "model_version": "0.0.0-test",
           "calibration_scope": "TEST §8 SCOPE STRING", "license": "GPL-3.0-or-later",
           "citation": "AlignAIR", "source_model_sha256": model_sha256}
    card.update(card_overrides or {})
    return save_aa_bundle(str(path), state_dict=built["model"].state_dict(), cfg=built["cfg"],
                          class_maps=built["class_maps"], d_allele_order=built["d_allele_order"],
                          asc_multi=built["asc_multi"], calibration=cal, card_extra=card)


def _rewrite_header(path, out_path, mutate) -> None:
    """Rebuild the container with every section payload byte-identical but the header mutated — the
    section index (and therefore `container.read_section`'s own per-section checksum) stays internally
    consistent, so only a header-vs-content cross-check (not a raw section-checksum failure) can catch
    the mutation. Used to simulate a header field going stale independent of its section (e.g. a
    tampered `class_counts` that no longer matches the still-correct, still-intact `class_maps` section)."""
    header = container.read_header(str(path))
    header.pop("_sections_base", None)
    names = list(header["sections"].keys())
    payloads = {name: container.read_section(str(path), name) for name in names}
    codecs = {name: header["sections"][name]["codec"] for name in names}
    mutate(header)
    sections = {name: (payloads[name], codecs[name]) for name in names}
    container.write_container(str(out_path), header, sections)


def _rewrite_calibration_section(path, out_path, new_calibration: dict) -> None:
    """Replace the `calibration` section's payload with `new_calibration` VERBATIM (bypassing
    `save_aa_bundle`'s `complete_calibration`, which would otherwise auto-fill a missing `method`
    block), keeping the header's own duplicate `integrity.calibration_sha256` claim in sync -- so only
    the loader's calibration-COMPLETENESS check (not a checksum/integrity-manifest check) can catch
    what is missing."""
    header = container.read_header(str(path))
    header.pop("_sections_base", None)
    names = list(header["sections"].keys())
    payloads = {name: container.read_section(str(path), name) for name in names}
    codecs = {name: header["sections"][name]["codec"] for name in names}
    new_bytes = canonical_json(new_calibration)
    payloads["calibration"] = new_bytes
    header["integrity"]["calibration_sha256"] = container.sha256_hex(new_bytes)
    sections = {name: (payloads[name], codecs[name]) for name in names}
    container.write_container(str(out_path), header, sections)


def _flip_byte_in_section(path, out_path, section_name) -> None:
    header = container.read_header(str(path))
    s = header["sections"][section_name]
    base = header["_sections_base"]
    data = bytearray(open(path, "rb").read())
    offset = base + s["offset"]
    data[offset] ^= 0xFF
    with open(out_path, "wb") as f:
        f.write(bytes(data))


# ------------------------------------------------------------------ (a) real v2 bundle


def test_resolve_path_hard_coded_ids_and_raw_path(tmp_path):
    assert str(_resolve_path(_REAL_MODEL_ID)) == str(_MODEL_MAP[_REAL_MODEL_ID])
    assert str(_resolve_path("alignair-aa-igh-human")) == str(_MODEL_MAP["alignair-aa-igh-human"])
    p = tmp_path / "x.alignair"
    p.write_bytes(b"whatever")
    assert _resolve_path(str(p)) == str(p)
    with pytest.raises(ValueError, match="unknown AA model id"):
        _resolve_path("no-such-model")


def test_load_real_v2_bundle_returns_predictor_with_working_forward_pass():
    pred = load_aa_model(_REAL_MODEL_ID)
    assert isinstance(pred, AAPredictor)

    # card + class maps + asc reconstruct and agree with each other
    assert pred.card["model_id"] == "alignair-aa-igh-human"
    assert pred.card["model_version"] == "2.0.0"
    assert pred.cfg.v_allele_count == pred.class_maps["V"].n_classes() == pred.card["class_counts"]["v"]
    assert pred.cfg.j_allele_count == pred.class_maps["J"].n_classes() == pred.card["class_counts"]["j"]
    assert pred.cfg.d_allele_count == len(pred.d_allele_order) == pred.card["class_counts"]["d"]
    assert isinstance(pred.asc_multi, dict) and len(pred.asc_multi) > 0

    # weights actually loaded: a forward pass runs and produces the right-shaped class logits
    tokens, _ = pad_tokenize_aa(["CARDLGYYYGMDVW", "CAKDRGYSSGWYFDYW"])
    tokens = torch.nn.functional.pad(tokens, (0, pred.cfg.max_seq_length - tokens.shape[1]))
    out = pred.model({"tokenized_sequence": tokens.to(pred.device)})
    assert out["v_class_logits"].shape == (2, pred.cfg.v_allele_count)
    assert out["j_class_logits"].shape == (2, pred.cfg.j_allele_count)
    assert out["d_class_logits"].shape == (2, pred.cfg.d_allele_count)

    # FrozenConformal V pooled qhat matches the embedded calibration exactly
    calibration = json.loads(container.read_section(str(_MODEL_MAP[_REAL_MODEL_ID]), "calibration"))
    v_frozen = calibration["genes"]["V"]["calibration_frozen"]
    assert pred.conformal["V"]["pooled"].qhat == pytest.approx(v_frozen["q_pooled"])
    for group, q in v_frozen["q_mondrian"].items():
        assert pred.conformal["V"]["mondrian"][group].qhat == pytest.approx(q)
    assert set(pred.conformal) == {"V", "J", "D"}


# ------------------------------------------------------------------ happy path on a fresh tmp bundle


def test_load_freshly_built_bundle_round_trips_and_binds(tmp_path, built):
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)

    pred = load_aa_model(str(path))
    assert isinstance(pred, AAPredictor)
    assert pred.cfg.v_allele_count == built["class_maps"]["V"].n_classes()
    assert pred.d_allele_order == built["d_allele_order"]
    assert set(pred.conformal) == {"V", "J", "D"}
    assert pred.conformal["V"]["pooled"].qhat == pytest.approx(0.9)
    assert pred.conformal["V"]["mondrian"]["short"].qhat == pytest.approx(0.95)

    tokens, _ = pad_tokenize_aa(["CARDLGYYYGMDVW"])
    tokens = torch.nn.functional.pad(tokens, (0, pred.cfg.max_seq_length - tokens.shape[1]))
    out = pred.model({"tokenized_sequence": tokens.to(pred.device)})
    assert out["v_class_logits"].shape == (1, pred.cfg.v_allele_count)


def test_load_rejects_legacy_pt_without_trust_pickle(tmp_path, built):
    path = tmp_path / "legacy.pt"
    torch.save({"model": built["model"].state_dict(), "cfg": built["cfg"].__dict__, "step": 1}, path)
    with pytest.raises(ValueError, match="trust_pickle"):
        load_aa_model(str(path))


def test_load_legacy_pt_with_trust_pickle_builds_a_working_model(tmp_path, built):
    path = tmp_path / "legacy.pt"
    torch.save({"model": built["model"].state_dict(), "cfg": built["cfg"].__dict__, "step": 1}, path)
    pred = load_aa_model(str(path), trust_pickle=True)
    assert isinstance(pred, AAPredictor)
    assert pred.class_maps == {}
    assert pred.conformal == {}
    tokens, _ = pad_tokenize_aa(["CARDLGYYYGMDVW"])
    tokens = torch.nn.functional.pad(tokens, (0, pred.cfg.max_seq_length - tokens.shape[1]))
    out = pred.model({"tokenized_sequence": tokens.to(pred.device)})
    assert out["v_class_logits"].shape == (1, pred.cfg.v_allele_count)


# ------------------------------------------------------------------ (b) binding-failure tests


def test_tamper_flip_byte_in_class_maps_section_raises(tmp_path, built):
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    tampered = tmp_path / "tampered_class_maps.alignair"
    _flip_byte_in_section(path, tampered, "class_maps")
    with pytest.raises(ValueError, match="checksum"):
        load_aa_model(str(tampered))


def test_tamper_flip_byte_in_weights_section_raises(tmp_path, built):
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    tampered = tmp_path / "tampered_weights.alignair"
    _flip_byte_in_section(path, tampered, "weights")
    with pytest.raises(ValueError, match="checksum"):
        load_aa_model(str(tampered))


def test_tamper_flip_byte_in_calibration_section_raises(tmp_path, built):
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    tampered = tmp_path / "tampered_calibration.alignair"
    _flip_byte_in_section(path, tampered, "calibration")
    with pytest.raises(ValueError, match="checksum"):
        load_aa_model(str(tampered))


def test_header_class_map_sha256_stale_relative_to_intact_section_raises(tmp_path, built):
    """A header integrity field that no longer matches its (untouched, self-consistent) section —
    catches a header-level edit that a raw per-section checksum re-check (see the tamper tests above)
    cannot, since the section's own index hashes were left internally consistent."""
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    stale = tmp_path / "stale_header.alignair"
    _rewrite_header(path, stale,
                    lambda h: h["reference_provenance"].__setitem__("class_map_sha256", "0" * 64))
    with pytest.raises(ValueError, match="integrity: class_map_sha256"):
        load_aa_model(str(stale))


def test_wrong_calibration_model_sha256_binding_raises(tmp_path, built):
    """The calibration's recorded model_sha256 differs from the bundle's source_model_sha256 (i.e. the
    calibration was fit against a DIFFERENT checkpoint) -> rejected as a binding failure."""
    path = tmp_path / "wrong_model.alignair"
    cal = _full_calibration(built["class_maps"], built["d_allele_order"], model_sha256=_OTHER_MODEL_SHA256)
    _write_bundle(built, path, model_sha256=_FAKE_MODEL_SHA256, calibration=cal)
    with pytest.raises(ValueError, match="calibration<->model binding.*model_sha256"):
        load_aa_model(str(path))


def test_wrong_calibration_class_map_db_hash_binding_raises(tmp_path, built):
    path = tmp_path / "wrong_db_hash.alignair"
    cal = _full_calibration(built["class_maps"], built["d_allele_order"], model_sha256=_FAKE_MODEL_SHA256)
    cal["provenance"]["class_map_db_hash"]["V"] = "0000000000000000"
    _write_bundle(built, path, model_sha256=_FAKE_MODEL_SHA256, calibration=cal)
    with pytest.raises(ValueError, match="calibration<->model binding.*class_map_db_hash"):
        load_aa_model(str(path))


def test_calibration_missing_method_block_raises(tmp_path, built):
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    cal = json.loads(container.read_section(str(path), "calibration"))
    del cal["method"]                     # save_aa_bundle would normally auto-fill this; bypass it
    bad = tmp_path / "no_method.alignair"
    _rewrite_calibration_section(path, bad, cal)
    with pytest.raises(ValueError, match="calibration incomplete: method"):
        load_aa_model(str(bad))


def test_calibration_missing_q_mondrian_raises(tmp_path, built):
    cal = _full_calibration(built["class_maps"], built["d_allele_order"], model_sha256=_FAKE_MODEL_SHA256)
    del cal["genes"]["J"]["calibration_frozen"]["q_mondrian"]
    path = tmp_path / "incomplete_qmondrian.alignair"
    _write_bundle(built, path, model_sha256=_FAKE_MODEL_SHA256, calibration=cal)
    with pytest.raises(ValueError, match="calibration incomplete: genes.J.calibration_frozen.q_mondrian"):
        load_aa_model(str(path))


def test_missing_binding_anchors_on_both_sides_raises(tmp_path, built):
    """Both `header.reference_provenance` and `calibration.provenance` have their binding-anchor
    fields (`source_model_sha256` / `class_map_db_hash`) deleted outright. A naive `left == right`
    binding check is vacuously satisfied when both sides are `None`; the loader must instead require
    BOTH sides to be present (non-empty) before ever comparing them, so deleting the anchors is itself
    a rejection, not a silent bypass."""
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)

    cal = json.loads(container.read_section(str(path), "calibration"))
    cal["provenance"] = {"supported_distribution": cal["provenance"]["supported_distribution"]}
    stripped_cal = tmp_path / "stripped_cal.alignair"
    _rewrite_calibration_section(path, stripped_cal, cal)

    final = tmp_path / "stripped_both.alignair"

    def _strip(h):
        h["reference_provenance"].pop("source_model_sha256", None)
        h["reference_provenance"].pop("class_map_db_hash", None)

    _rewrite_header(stripped_cal, final, _strip)

    with pytest.raises(ValueError, match="calibration<->model binding.*missing"):
        load_aa_model(str(final))


def test_fabricated_but_mutually_matching_class_map_db_hash_raises(tmp_path, built):
    """Header and calibration both claim the SAME fabricated `class_map_db_hash` -- they agree with
    each other, but neither matches the real, checksum-verified `class_maps[gene].db_hash`. The
    binding must anchor on the real class maps, not on two mutually-consistent header/calibration
    duplicates."""
    fake = {"V": "ffffffffffffffff", "J": "ffffffffffffffff"}
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)

    real_v_hash = built["class_maps"]["V"].db_hash
    assert real_v_hash != fake["V"]

    cal = json.loads(container.read_section(str(path), "calibration"))
    cal["provenance"]["class_map_db_hash"] = dict(fake)
    faked_cal = tmp_path / "faked_cal.alignair"
    _rewrite_calibration_section(path, faked_cal, cal)

    final = tmp_path / "faked_both.alignair"
    _rewrite_header(faked_cal, final,
                    lambda h: h["reference_provenance"].__setitem__("class_map_db_hash", dict(fake)))

    with pytest.raises(ValueError, match="calibration<->model binding.*class_map_db_hash"):
        load_aa_model(str(final))


def test_missing_model_sha256_on_header_side_only_raises(tmp_path, built):
    """Calibration still carries `provenance.model_sha256`; only the header's
    `reference_provenance.source_model_sha256` is deleted -- a present-vs-absent mismatch, not a
    present-vs-present value mismatch, must still be rejected."""
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    stripped = tmp_path / "stripped_header_model_sha.alignair"
    _rewrite_header(path, stripped,
                    lambda h: h["reference_provenance"].pop("source_model_sha256", None))
    with pytest.raises(ValueError, match="calibration<->model binding.*model_sha256 missing"):
        load_aa_model(str(stripped))


def test_config_sha256_stale_relative_to_intact_section_raises(tmp_path, built):
    """The header's `integrity.config_sha256` no longer matches the (untouched, self-consistent)
    `config` section -- the integrity manifest must independently recompute and check this hash, not
    just rely on `read_section`'s own per-section checksum."""
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    stale = tmp_path / "stale_config.alignair"
    _rewrite_header(path, stale,
                    lambda h: h["integrity"].__setitem__("config_sha256", "0" * 64))
    with pytest.raises(ValueError, match="integrity: config_sha256"):
        load_aa_model(str(stale))


def test_label_count_mismatch_raises(tmp_path, built):
    """The header's `class_counts.v` disagrees with the (untouched, correct) class_maps section and
    model head -- simulates a calibration/bundle assembled against a different label-count reference."""
    path = tmp_path / "ok.alignair"
    _write_bundle(built, path)
    bad = tmp_path / "bad_counts.alignair"
    _rewrite_header(path, bad,
                    lambda h: h["class_counts"].__setitem__("v", h["class_counts"]["v"] + 1))
    with pytest.raises(ValueError, match="V label-count mismatch"):
        load_aa_model(str(bad))
