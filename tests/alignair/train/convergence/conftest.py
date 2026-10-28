"""Hand-written convergence scoring inputs shared by metric and preflight contracts."""
from copy import deepcopy

import pytest

from alignair.core import AlignAIR, build_model_config
from alignair.data.projection import CODON_TABLE
from alignair.reference.labels import EquivalenceClassMap


@pytest.fixture
def evaluation_inputs():
    class_maps = {g: EquivalenceClassMap(g, "fixture", {g+"0*01": 0, g+"1*01": 1},
                                       {0: [g+"0*01"], 1: [g+"1*01"]}) for g in ("V", "J")}
    cfg = build_model_config(2, 2, 2, input_space='protein', label_space='protein_classes',
                             max_seq_length=128, has_d=True)
    meta = {"d_index": {"D0*01": 0, "D1*01": 1}}
    codons = {aa: codon for codon, aa in CODON_TABLE.items()}
    sequences = ["ACDEFGHIKLMNPQRSTVWY", "YWVTSRQPNMLKIHGFEDCA"]
    def row(sequence, index):
        calls = {g.lower()+"_call": g+str(index)+"*01" for g in ("V", "D", "J")}
        record = {"sequence": "".join(codons[aa] for aa in sequence), "sequence_aa": sequence,
                  "productive": True, "junction_start": 24, **calls,
                  "v_sequence_start": 0, "v_sequence_end": 24,
                  "d_sequence_start": 24, "d_sequence_end": 48,
                  "j_sequence_start": 48, "j_sequence_end": 60}
        return {"aa": sequence, **calls, "d_inverted": False, "record": record}
    clean = [row(s, i) for i, s in enumerate(sequences)]
    strata = {name: deepcopy(clean) for name in ("clean", "aa_truncate_50_80", "short_janchor", "moderate_full")}
    for item in strata["aa_truncate_50_80"]:
        item["aa"] = item["aa"][:12]
        item.pop("record")
    for item in strata["short_janchor"]:
        item["aa"] = item["aa"][-8:]
        item.pop("record")
    calibration = [row(s[1:]+s[:1], i) for i, s in enumerate(sequences)]
    cohort = {"by_stratum": strata, "calibration": calibration, "dc_name": "fixture"}
    return AlignAIR(cfg).eval(), cfg, class_maps, meta, cohort
