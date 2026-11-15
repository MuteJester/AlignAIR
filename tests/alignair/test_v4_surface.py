"""AlignAIR 4 has one prediction surface: no fixed-head contract, no genotype mask and no second germline reader."""
import contextlib
import dataclasses
import importlib.util
import inspect
import io
import tomllib
from pathlib import Path

import pytest

from alignair.cli.main import main
from alignair.core import AlignAIR, ModelComponents
from alignair.predict.threshold import select_alleles

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]


def _gone(module):
    """No module of that name can be imported: no spec, or only the namespace a checkout's leftover ``__pycache__``
    keeps (git does not remove an ignored directory), with no module in it."""
    spec = importlib.util.find_spec(module)
    return spec is None or spec.origin is None and not any(
        path.suffix == ".py" or path.is_dir() and path.name != "__pycache__"
        for location in spec.submodule_search_locations for path in Path(location).iterdir())


def test_the_alignair_3_prediction_surface_is_gone():
    assert _gone("alignair.align") and _gone("alignair.reference.identity")
    assert "reference_agnostic" not in {field.name for field in dataclasses.fields(ModelComponents)}
    assert not hasattr(AlignAIR, "reference_agnostic")
    assert "allowed" not in inspect.signature(select_alleles).parameters
    out = io.StringIO()
    with contextlib.redirect_stdout(out), pytest.raises(SystemExit):
        main(["predict", "--help"])
    assert "--genotype-method" not in out.getvalue()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    declared = [*project["dependencies"], *(d for group in project["optional-dependencies"].values() for d in group)]
    assert not any(name in dependency for dependency in declared for name in ("parasail", "pywfa", "biopython"))


def test_prediction_has_one_forward_collector():
    """``run_model`` collects the forward's outputs; the generic collector and the output-key selection are gone."""
    import alignair.predict as predict
    from alignair.predict import forward
    assert not hasattr(forward, "collect_prediction_outputs")
    assert not {"forward_batch", "prediction_batches", "collect_prediction_outputs"} & set(predict.__all__)
    for function in (forward.forward_batch, forward.prediction_batches):
        assert "output_keys" not in inspect.signature(function).parameters


def test_allele_selection_has_one_rule_and_reads_are_right_padded():
    """Every model calls the alleles scoring at least the threshold (capped) and right-pads its reads: the AlignAIR 3
    selectors, the TF centre padding and the header block that restated them are gone."""
    from alignair.model_file import save_model
    from alignair.predict import PredictConfig, threshold
    from alignair.predict.segment import correct_segments
    assert not {"selector", "pad_mode"} & {field.name for field in dataclasses.fields(PredictConfig)}
    assert not any(hasattr(threshold, name) for name in ("SELECTORS", "largest_gap", "max_likelihood_percentage"))
    assert list(inspect.signature(select_alleles).parameters) == ["allele_probs", "names", "threshold", "cap"]
    assert not {"max_len", "pad_mode"} & set(inspect.signature(correct_segments).parameters)
    assert "inference" not in inspect.signature(save_model).parameters


def test_non_finite_outputs_are_refused_once_by_the_forward():
    """``forward_batch`` refuses a non-finite output (``NumericalError``); the pipeline's later checks of the same,
    which could never fire, are gone."""
    from alignair.predict import stages
    assert not hasattr(stages, "validate_predictions") and not hasattr(stages, "assert_finite")


def test_the_germline_reader_has_its_settings_fixed_and_returns_cigars():
    """The matcher is built with the settings every model aligns with; the CIGAR derivation returns the CIGAR (its
    result object only ever carried the CIGAR); the unused anchor decoder and gene tuple are gone."""
    from alignair.predict import decode, heuristic_matcher, state
    assert list(inspect.signature(heuristic_matcher.HeuristicGermlineMatcher).parameters) == ["germlines"]
    assert heuristic_matcher.derive_cigar("ACGGT", "ACGT", 1) == "2M1I2M"
    assert not hasattr(heuristic_matcher, "derive_alignment")
    assert not any(hasattr(state, name) for name in ("AlignResult", "GENES"))
    assert not hasattr(decode, "decoded_anchors")


def test_the_benchmark_package_imports_its_api_plainly():
    """``alignair.evaluate`` has no lazy-import shim (importing ``alignair`` already loads the prediction API)."""
    import alignair.evaluate as evaluate
    assert not hasattr(evaluate, "__getattr__")
    assert evaluate.run_benchmark.__module__ == "alignair.evaluate.benchmark"


def test_genotype_files_are_parsed_once_and_checked_once():
    """``load_genotype`` parses a file; ``genotype_bank`` is the one check of its alleles against a reference. The
    genotype study's per-gene call record and the always-empty V-confidence report are gone."""
    from alignair.analyze import report
    from alignair.genotype import airr, load_genotype
    assert list(inspect.signature(load_genotype).parameters) == ["path"]
    assert not hasattr(airr, "GeneCall")
    assert "v_set_confidence" not in report.qc([{"v_call": "IGHV1-2*01", "v_set_confidence": "0.9"}])


def test_the_model_package_keeps_no_unused_or_legacy_symbols():
    """No unused bank key list, no second way to construct the model, one region index space (no AlignAIR 3 ``pad``
    region), no trunk width that is not the configured one, and one allele encoder."""
    from alignair.core import regions
    from alignair.model_file import read_dataconfig
    from alignair.network import bank, chip, trunk
    assert not hasattr(bank, "EXTENSION_KEYS")
    assert not hasattr(AlignAIR, "from_config") and not hasattr(AlignAIR, "from_dataconfigs")
    assert inspect.signature(read_dataconfig).parameters["index"].default is inspect.Parameter.empty
    assert regions.REGIONS == ("pre", "V", "N1", "D", "N2", "J", "post") and not hasattr(regions, "READ_REGIONS")
    assert not hasattr(trunk, "TRUNK_CHANNELS") and list(inspect.signature(chip.ChipCore).parameters) == ["cfg"]
    assert _gone("alignair.data.encoders")


def test_the_switches_every_trained_model_used_one_way_are_retired():
    """The orientation head, the anchor heads and (with a D) the D presence head are always built, and an empty read
    is never refused for being empty; ``bank_orientation`` stays (it adds no weights)."""
    from alignair.core.config import AlignAIRConfig
    from alignair.data.batching import collate_samples
    fields = {field.name for field in dataclasses.fields(AlignAIRConfig)}
    assert not {"use_orientation", "anchor_heads", "d_presence_head", "reject_empty_sequences"} & fields
    assert "bank_orientation" in fields
    assert "reject_empty_sequences" not in inspect.signature(collate_samples).parameters


def test_reads_have_one_nucleotide_normaliser():
    """A read is normalised by one rule, for a file (dropped when mostly unusable) or a prediction (cut to the
    window); the generic, protein-era validation framework and its compatibility names are gone."""
    from alignair.data import inputs
    from alignair.io import sequence_reader
    gone = ("SequenceValidationPolicy", "normalize_sequence_inputs", "NormalizedSequenceBatch", "SequenceInputFailure",
            "SequenceRecord", "nucleotide_file_policy", "nucleotide_prediction_policy", "validate_nucleotide_sequence",
            "validate_nucleotide")
    assert not any(hasattr(inputs, name) for name in gone)
    assert inputs.normalize_file_read(" acgtr ").sequence == "ACGTN"
    assert not hasattr(sequence_reader, "validate") and not hasattr(sequence_reader, "validate_sequence")
