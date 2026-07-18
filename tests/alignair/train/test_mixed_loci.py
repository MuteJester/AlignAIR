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

from alignair.train.build import required_segments
from alignair.train.gym import Curriculum, build_experiment
from alignair.train.trainer import (_amplicon_specs, _cap_for, _dc_name, _effective_mutation_caps,
                                    _is_tcr)

_IGH = gd.HUMAN_IGH_OGRDB
_TRA = gd.HUMAN_TCRA_IMGT
_TRB = gd.HUMAN_TCRB_IMGT

_PROGRESSES = (0.3, 0.6, 0.9)
_HEAVY_SHM = 0.25


def _rates(dc, mutation_cap=None):
    """The distinct SHM rates every stream of this locus would actually be simulated with."""
    specs = _amplicon_specs(_PROGRESSES, _HEAVY_SHM, 1, _cap_for(dc, mutation_cap))
    return sorted({round(s["mutation_rate"], 4) for s in specs})


# --- IG + TCR: the SHM cap must be per locus, never global ------------------------------------

def test_is_tcr_classifies_loci():
    assert _is_tcr(_TRA) and _is_tcr(_TRB)
    assert not _is_tcr(_IGH)


def test_tcr_loci_are_always_capped_to_zero_and_ig_is_untouched():
    assert _cap_for(_TRA, None) == 0.0
    assert _cap_for(_TRB, None) == 0.0
    assert _cap_for(_IGH, None) is None            # IG keeps the run-level cap (uncapped by default)


def test_mixed_ig_tcr_keeps_shm_for_ig_and_zeroes_tcr():
    """The exact bug: in a combined IGH+TRB run, IGH must keep its SHM curriculum."""
    igh_rates, trb_rates = _rates(_IGH), _rates(_TRB)
    assert max(igh_rates) > 0, "IGH lost its SHM curriculum in a mixed IG+TCR run"
    assert _HEAVY_SHM in igh_rates, "IGH lost its heavy-SHM stream"
    assert trb_rates == [0.0], f"TCR streams must have zero SHM, got {trb_rates}"


def test_a_global_cap_would_have_broken_ig_shm():
    """Guards the *reason* the cap is per-locus: a global 0.0 flattens IGH's whole ladder."""
    global_capped = sorted({round(s["mutation_rate"], 4)
                            for s in _amplicon_specs(_PROGRESSES, _HEAVY_SHM, 1, 0.0)})
    assert global_capped == [0.0]
    assert _rates(_IGH) != global_capped           # the per-locus path must NOT behave that way


def test_tcr_streams_never_request_shm_from_genairr():
    """GenAIRR raises if mutate() is called on a TCR refdata, so every TCR spec must be rate 0."""
    for dc in (_TRA, _TRB):
        specs = _amplicon_specs(_PROGRESSES, _HEAVY_SHM, 1, _cap_for(dc, None))
        assert all(s["mutation_rate"] == 0.0 for s in specs)


# --- provenance: the card must record the caps that were ACTUALLY applied ----------------------

def test_dc_name_resolves_canonical_genairr_names():
    assert _dc_name(_TRA) == "HUMAN_TCRA_IMGT"
    assert _dc_name(_IGH) == "HUMAN_IGH_OGRDB"


def test_effective_caps_single_tcr_locus_records_the_zero():
    """A TRA run must not read as `mutation_cap: null` (uncapped) when it saw zero SHM."""
    assert _effective_mutation_caps([_TRA], None) == {"HUMAN_TCRA_IMGT": 0.0}


def test_effective_caps_mixed_run_records_each_locus():
    caps = _effective_mutation_caps([_IGH, _TRB], None)
    assert caps == {"HUMAN_IGH_OGRDB": None, "HUMAN_TCRB_IMGT": 0.0}


def test_effective_caps_disambiguates_repeated_dataconfigs():
    caps = _effective_mutation_caps([_IGH, _IGH], None)
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
    params = dict(Curriculum().params(0.3))
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
