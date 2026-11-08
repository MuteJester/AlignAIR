"""The prediction pipeline is an ordered list of named stages over one state (``predict.pipeline``): ``predict`` is
the list run without hooks, a hook wraps every stage call (the first hook outermost) and may replace what a stage
returns, a trace hook captures each stage's output state and seconds, and a stage that leaves a field it sets unset
is refused."""
import dataclasses
import json

import pytest

from alignair.errors import InferenceError
from alignair.predict import PredictConfig
from alignair.predict.pipeline import PIPELINE, STAGE_NAMES, Stage, Trace, predict, run_stages
from alignair.predict.state import GeneCall, PredictionState
from tests.alignair.predict.golden_rows import FIXTURE, golden_model

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def golden():
    model, reference = golden_model()
    return model, reference, json.loads(FIXTURE.read_text())["reads"]


def _start(model, reference, reads):
    return PredictionState.start(model, reference, PredictConfig.for_model(model, reference, batch_size=8), "cpu",
                                 reads)


def test_the_pipeline_is_the_stages_in_their_order():
    assert STAGE_NAMES == ("input_policy", "forward", "clean", "canonicalize", "segments", "alleles", "germline",
                           "better_v", "v_presence", "ties", "records", "airr")
    assert [stage.sets for stage in PIPELINE] == [("reads", "cropped"), ("batches",), ("predictions",),
                                                  ("canonical",), ("segments",), ("calls",), ("alignments",),
                                                  ("alignments",), ("no_v",), ("v_ties",), ("records",), ("rows",)]


def test_predict_is_the_pipeline_run_without_hooks(golden):
    from alignair.api import predict_sequences
    model, reference, reads = golden
    cfg = PredictConfig.for_model(model, reference, batch_size=8)
    trace = Trace()
    traced = run_stages(PredictionState.start(model, reference, cfg, "cpu", reads), hooks=(trace,))
    assert traced.rows == predict(model, reads, reference, cfg, "cpu") == predict_sequences(
        model, reference, reads, device="cpu", batch_size=8)
    assert list(trace.seconds) == list(STAGE_NAMES) and min(trace.seconds.values()) >= 0
    assert all(getattr(trace.states[stage.name], name) is not None for stage in PIPELINE for name in stage.sets)


def test_hooks_nest_in_order_and_may_replace_a_stages_output(golden):
    model, reference, reads = golden
    pick = reference.gene("V").names[7]
    seen = []

    def outer(stage, state, run):
        seen.append(("outer", stage.name))
        return run(state)

    def inner(stage, state, run):
        seen.append(("inner", stage.name))
        out = run(state)
        if stage.name != "alleles":
            return out
        return dataclasses.replace(out, calls={**out.calls, "v": [GeneCall((pick,), (1.0,)), *out.calls["v"][1:]]})

    rows = run_stages(_start(model, reference, reads), hooks=(outer, inner)).rows
    assert seen[:2] == [("outer", "input_policy"), ("inner", "input_policy")]
    assert rows[0]["v_calls"] == [pick]


def test_a_stage_that_leaves_its_output_unset_is_refused(golden):
    model, reference, reads = golden
    lazy = (*PIPELINE[:5], Stage("alleles", lambda state: state, ("calls",)), *PIPELINE[6:])
    with pytest.raises(InferenceError, match="alleles left calls unset"):
        run_stages(_start(model, reference, reads), lazy)
