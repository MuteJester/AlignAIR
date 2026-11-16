"""Regression guards for MIXED-locus training invariants (IG+TCR, and D + non-D).

These pin behaviour that is easy to break with a single "global" decision derived from the union of
dataconfigs. Both bugs below were real:

  * a run-level ``0.0 if any(TCR)`` mutation cap disabled SHM for IGH in a combined IGH+TRB model;
  * a union ``ref.has_d`` dropped every TRA record from a TRA+TRB validation report.

They are cheap: no training, no checkpoints — just the stream/param builders.
"""
import itertools

import GenAIRR.data as gd
import pytest

from alignair.train.bundle import required_segments
from alignair.genairr.experiment import build_experiment
from alignair.train.records import Curriculum
from alignair.genairr.catalog import dataconfig_name
from alignair.train.records.streams import (CurriculumStreamPolicy, mutation_cap_for, effective_mutation_caps)
from alignair.genairr.catalog import is_tcr


pytestmark = pytest.mark.integration

_IGH = gd.HUMAN_IGH_OGRDB
_TRA = gd.HUMAN_TCRA_IMGT
_TRB = gd.HUMAN_TCRB_IMGT

_PROGRESSES = (0.3, 0.6, 0.9)
_HEAVY_SHM = 0.25


def _rates(dc):
    """The distinct SHM rates every stream of this locus would actually be simulated with."""
    specs = CurriculumStreamPolicy(_PROGRESSES, _HEAVY_SHM, 1).parameters_for(dc)
    return sorted({round(s["mutation_rate"], 4) for s in specs})


# --- IG + TCR: the SHM cap must be per locus, never global ------------------------------------

def test_is_tcr_classifies_loci():
    assert is_tcr(_TRA) and is_tcr(_TRB)
    assert not is_tcr(_IGH)


def test_tcr_loci_are_always_capped_to_zero_and_ig_is_untouched():
    assert mutation_cap_for(_TRA) == 0.0
    assert mutation_cap_for(_TRB) == 0.0
    assert mutation_cap_for(_IGH) is None                  # IG is uncapped


def test_mixed_ig_tcr_keeps_shm_for_ig_and_zeroes_tcr():
    """The exact bug: in a combined IGH+TRB run, IGH must keep its SHM curriculum."""
    igh_rates, trb_rates = _rates(_IGH), _rates(_TRB)
    assert max(igh_rates) > 0, "IGH lost its SHM curriculum in a mixed IG+TCR run"
    assert _HEAVY_SHM in igh_rates, "IGH lost its heavy-SHM stream"
    assert trb_rates == [0.0], f"TCR streams must have zero SHM, got {trb_rates}"


def test_the_tcr_zero_cap_leaves_the_ig_shm_ladder_whole():
    """Guards the *reason* the cap is per locus: a global 0.0 would flatten IGH's whole ladder."""
    assert _rates(_TRB) == [0.0]
    assert len(_rates(_IGH)) > 1 and 0.0 not in _rates(_IGH)


def test_tcr_streams_never_request_shm_from_genairr():
    """GenAIRR raises if mutate() is called on a TCR refdata, so every TCR spec must be rate 0."""
    for dc in (_TRA, _TRB):
        specs = CurriculumStreamPolicy(_PROGRESSES, _HEAVY_SHM).parameters_for(dc)
        assert all(s["mutation_rate"] == 0.0 for s in specs)


# --- provenance: the card must record the caps that were ACTUALLY applied ----------------------

def test_dc_name_resolves_canonical_genairr_names():
    assert dataconfig_name(_TRA) == "HUMAN_TCRA_IMGT"
    assert dataconfig_name(_IGH) == "HUMAN_IGH_OGRDB"


def test_effective_caps_single_tcr_locus_records_the_zero():
    """A TRA run must not read as `mutation_cap: null` (uncapped) when it saw zero SHM."""
    assert effective_mutation_caps([_TRA]) == {"HUMAN_TCRA_IMGT": 0.0}


def test_effective_caps_mixed_run_records_each_locus():
    caps = effective_mutation_caps([_IGH, _TRB])
    assert caps == {"HUMAN_IGH_OGRDB": None, "HUMAN_TCRB_IMGT": 0.0}


def test_effective_caps_disambiguates_repeated_dataconfigs():
    caps = effective_mutation_caps([_IGH, _IGH])
    assert len(caps) == 2, f"a repeated dataconfig must not overwrite its twin: {caps}"


# --- D + non-D: required segments must come from each locus, not the union ---------------------

def test_required_segments_are_per_locus():
    assert required_segments(_TRA) == ("v", "j")          # no D gene
    assert required_segments(_TRB) == ("v", "d", "j")
    assert required_segments(_IGH) == ("v", "d", "j")


@pytest.mark.parametrize("dc,name", [(_TRA, "TRA"), (_TRB, "TRB")])
def test_mixed_d_and_nond_validation_retains_every_locus(dc, name):
    """TRA+TRB: a union has_d would drop ALL TRA records (they have no D coordinates), silently
    reducing the validation report to TRB. Per-locus segments must retain both."""
    params = {**Curriculum().params(0.3), "invert_d_prob": 0.05}
    params["mutation_rate"] = 0.0                          # TCR: no SHM
    exp = build_experiment(dc, params, allow_curatable=True)
    recs = list(itertools.islice(exp.stream_records(n=None, seed=7), 8))
    assert recs

    kept = [r for r in recs
            if all(r.get(f"{g}_sequence_start") is not None for g in required_segments(dc))]
    assert len(kept) == len(recs), f"{name}: per-locus filter dropped records"

    if name == "TRA":   # the union rule ("v","d","j") is exactly what used to drop every TRA record
        union_kept = [r for r in recs
                      if all(r.get(f"{g}_sequence_start") is not None for g in ("v", "d", "j"))]
        assert union_kept == [], "TRA records unexpectedly carry D coords; the guard is not meaningful"
