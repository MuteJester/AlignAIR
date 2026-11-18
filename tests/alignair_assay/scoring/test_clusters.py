"""Cluster ids: the reads a paired comparison must not treat as independent (a clone's members, a read's two
strands)."""
from dataclasses import replace

import pytest

from alignair_assay.scoring.clusters import CLUSTER_KEYS, cluster_ids
from alignair_assay.truth.model import ObservableTruth

pytestmark = pytest.mark.component


def _truth(case_id: str, stratum: str, groups: dict) -> ObservableTruth:
    return ObservableTruth(case_id=case_id, stratum=stratum, sequence="ACGT", orientation=0, segments={},
                           unscorable=frozenset(), junction=None, vj_in_frame=None, stop_codon=None, productive=None,
                           clone_key=None, v_mismatches=None, v_frameshift=None, j_frameshift=None, excluded=None,
                           warnings=(), groups=groups)


def test_cluster_ids_name_the_first_key_present_within_its_stratum():
    truths = [_truth("S9:lineages:000000", "lineages", {"clone": "3"}),
              _truth("S9:lineages:000001", "lineages", {"clone": "3"}),
              _truth("S9:other:000000", "other", {"clone": "3"}),  # clone ids are per GenAIRR run (per stratum)
              _truth("S11:full_igg:000000:fwd", "full_igg", {"pair": "S11:full_igg:000000"}),
              _truth("x", "s", {"clone": "1", "pair": "p"}),  # a clone contains its pairs: the coarser key wins
              _truth("S1:cohort:000000", "cohort", {"subject": "donor1"}),  # not a default key
              _truth("plain", "s", {})]
    assert CLUSTER_KEYS == ("clone", "pair")
    assert cluster_ids(truths) == {"S9:lineages:000000": "lineages/clone:3", "S9:lineages:000001": "lineages/clone:3",
                                   "S9:other:000000": "other/clone:3",
                                   "S11:full_igg:000000:fwd": "full_igg/pair:S11:full_igg:000000",
                                   "x": "s/clone:1"}
    # Other keys are opt-in, in the order given.
    assert cluster_ids(truths, keys=("subject",)) == {"S1:cohort:000000": "cohort/subject:donor1"}
    assert cluster_ids([replace(truths[4], groups={"pair": "p"})], keys=("subject", "pair")) == {"x": "s/pair:p"}


def test_the_suites_that_group_reads_emit_the_cluster_keys():
    # suites.engine gives "clone" to the cases of a clonal stratum and "pair" to both strands of a
    # both-orientations stratum; S9 and S11 are those suites.
    pytest.importorskip("GenAIRR")
    from alignair_assay.suites import SUITES

    clonal = {suite.name for suite in SUITES.values() if any(stratum.clonal for stratum in suite.strata)}
    paired = {suite.name for suite in SUITES.values() if any(stratum.both_orientations for stratum in suite.strata)}
    assert clonal == {"S9"} and paired == {"S11"}
