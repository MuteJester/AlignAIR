"""Genotype task 2: repertoire alignment + weighted usage aggregation."""
import numpy as np

from alignair.genotype.aggregate import AlignedRepertoire, weighted_usage


def _stub():
    # 2 reads, 3 V alleles. read0: confident V0, low SHM, long; read1: confident V1, high SHM, short.
    return AlignedRepertoire(
        sequences=["A" * 300, "A" * 100],
        allele_probs={"v": np.array([[0.9, 0.05, 0.05], [0.05, 0.9, 0.05]])},
        records=[{}, {}],
        mutation_rate=np.array([0.02, 0.30]),
        read_lengths=np.array([300, 100]),
        state_logits=None,
        gene_names={"v": ["V0", "V1", "V2"]},
        reference=None,
    )


def test_weighted_usage_upweights_reliable_reads():
    u = weighted_usage(_stub(), "v")
    assert u["V0"]["mass"] > u["V1"]["mass"]          # equal probs, but V0's read is low-SHM + long
    assert u["V0"]["count"] == 1 and u["V1"]["count"] == 1 and u["V2"]["count"] == 0


def test_state_logits_field_optional():
    assert _stub().state_logits is None               # None when the model has no state head


def test_clean_surfaces_state_logits_when_present():
    from alignair.predict.clean import clean
    B, L, S = 2, 4, 4
    batch = {"v_allele": np.zeros((B, 3)), "v_start": np.zeros(B), "v_end": np.zeros(B),
             "j_allele": np.zeros((B, 2)), "j_start": np.zeros(B), "j_end": np.zeros(B),
             "productive": np.zeros(B), "mutation_rate": np.zeros(B), "indel_count": np.zeros(B),
             "state_logits": np.zeros((B, L, S))}
    preds = clean([batch], ("v", "j"))
    assert preds.state_logits is not None and preds.state_logits.shape == (B, L, S)
    # absent -> None
    del batch["state_logits"]
    assert clean([batch], ("v", "j")).state_logits is None
