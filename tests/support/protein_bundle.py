"""Generated protein bundles shared by loader and registry integration tests."""
import GenAIRR as ga

from alignair.predict.protein.bundle import save_aa_bundle
from alignair.reference import asc_to_imgt_multi
from tests.support.model_setup import protein_class_setup

MODEL_SHA256 = "cafebabe" * 8


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


def protein_calibration(class_maps, d_allele_order, *, model_sha256):
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


def build_protein_bundle_setup(*, has_d=True):
    """Build reference-bound test weights once per consuming module fixture."""
    setup = protein_class_setup("HUMAN_IGH_OGRDB", 192, has_d=has_d)
    model = setup.build_model()
    cfg, class_maps = setup.config, setup.protein_reference.class_maps
    meta = {'d_index': dict(setup.reference.gene('D').index) if setup.config.has_d else {}}
    dc = ga.HUMAN_IGH_OGRDB
    asc_multi = asc_to_imgt_multi(dc)
    d_allele_order = [name for name, _ in sorted(meta["d_index"].items(), key=lambda kv: kv[1])]
    return {"model": model, "cfg": cfg, "class_maps": class_maps, "d_allele_order": d_allele_order,
            "asc_multi": asc_multi}


def write_protein_bundle(built, path, *, model_sha256=MODEL_SHA256, calibration=None,
                         card_overrides=None):
    cal = calibration if calibration is not None else protein_calibration(
        built["class_maps"], built["d_allele_order"], model_sha256=model_sha256)
    card = {"model_id": "alignair-aa-igh-human-test", "model_version": "0.0.0-test",
           "calibration_scope": "TEST §8 SCOPE STRING", "license": "GPL-3.0-or-later",
           "citation": "AlignAIR", "source_model_sha256": model_sha256}
    card.update(card_overrides or {})
    return save_aa_bundle(str(path), state_dict=built["model"].state_dict(), cfg=built["cfg"],
                          class_maps=built["class_maps"], d_allele_order=built["d_allele_order"],
                          asc_multi=built["asc_multi"], calibration=cal, card_extra=card)
