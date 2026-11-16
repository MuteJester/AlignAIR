"""AlignAIR 4 has one prediction surface: no fixed-head contract, no genotype mask and no second germline reader."""
import contextlib
import dataclasses
import importlib.util
import inspect
import io
from pathlib import Path

try:
    import tomllib  # Python >= 3.11
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

import pytest

import alignair
import alignair.train
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
    """The locator is built with the settings every model aligns with; the CIGAR derivation returns the CIGAR (its
    result object only ever carried the CIGAR); the unused anchor decoder and gene tuple are gone; the reader is one
    package (no heuristic_matcher module beside it)."""
    from alignair.predict import decode, state
    from alignair.predict.germline import cigar, locate
    assert list(inspect.signature(locate.GermlineLocator).parameters) == ["germlines"]
    assert cigar.derive_cigar("ACGGT", "ACGT", 1) == "2M1I2M"
    assert not hasattr(cigar, "derive_alignment")
    assert _gone("alignair.predict.heuristic_matcher") and _gone("alignair.predict.cigar")
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


def test_the_record_policy_keeps_no_field_every_model_set_one_way():
    """Every model trained with the automatic mix, the weighted indels and no mutation cap (a TCR reference is capped
    at zero on its own); ``train_args`` still record the three values, so they compare with earlier checkpoints'."""
    from alignair.train.records.streams import CurriculumStreamPolicy
    assert not {"mode", "indel_form", "mutation_cap"} & {field.name for field in dataclasses.fields(
        CurriculumStreamPolicy)}


def test_the_training_engine_has_no_extension_hooks():
    """Training steps AdamW under the run's ``CosineSchedule`` on the objective the model's components build; an
    architecture plugs in through ``ModelComponents``, not through optimizer, scheduler or task factories."""
    from alignair.train import engine, setup
    assert not hasattr(setup, "OptimizerFactory") and "OptimizerFactory" not in alignair.train.__all__
    assert list(inspect.signature(setup.build_training_engine).parameters) == ["model", "tasks", "device", "lr",
                                                                               "grad_clip", "schedule"]
    assert inspect.signature(engine.TrainingEngine).parameters["tasks"].default is inspect.Parameter.empty


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


def test_training_has_no_study_schedule():
    """``val_every`` and ``on_iteration`` do what the study schedule did; it, and the run fields only it filled, are
    gone."""
    import alignair.train as train
    from alignair.train import trainer
    from alignair.train.config import TrainingConfig
    from alignair.train.workflow import TrainingRun
    assert not hasattr(train, "TrainingStudy") and _gone("alignair.train.study")
    assert "study" not in {field.name for field in dataclasses.fields(TrainingConfig)}
    assert "study" not in inspect.signature(trainer._train).parameters
    assert {field.name for field in dataclasses.fields(TrainingRun)} == {"model_path", "best_model_path", "steps"}


def test_training_has_one_entry_point():
    """``run_training`` is the one way to train: ``train_model`` skipped its seeding and had defaults of its own."""
    from alignair import api
    from alignair.train import trainer
    assert not hasattr(alignair, "train_model") and not hasattr(api, "train_model")
    assert "train_model" not in alignair.__all__
    assert not hasattr(trainer, "train")             # its private _train takes the run's config and record policy


def test_training_names_live_in_one_module_each():
    """No compatibility re-exports: the trainer re-exports neither the batch builder, the validator, the task metrics
    nor the GenAIRR identity; ``alignair.train`` does not re-export the runtime's evaluation mode, nor the gym the
    targets builder; the study's parameters and the warm-start offset are gone; the training records are
    ``train.records`` (no ``train.gym``)."""
    import alignair.train as train
    from alignair.data import adapters
    from alignair.genairr import novel
    from alignair.tasks import build_tasks
    from alignair.train import checkpoints, engine, records, trainer
    assert not any(hasattr(trainer, name) for name in ("build_batch", "validate", "eval_metrics",
                                                       "genairr_source_sha256"))
    assert not hasattr(adapters, "build_nucleotide_batch")
    assert "evaluation_mode" not in train.__all__ and "build_targets" not in records.__all__
    assert _gone("alignair.train.gym") and "build_experiment" not in records.__all__
    assert "start_step" not in inspect.signature(trainer._train).parameters
    assert "cfg" not in inspect.signature(checkpoints.save_checkpoint).parameters
    assert "cfg" not in inspect.signature(checkpoints.save_rotating).parameters
    assert "cfg" not in inspect.signature(engine.validate).parameters
    assert "genes" not in inspect.signature(build_tasks).parameters
    assert "indels" not in inspect.signature(novel.derive_variant).parameters


def test_the_shipped_package_has_no_test_only_record_stream():
    """``ParallelRecords`` (``training_records`` over worker processes, which only tests built) lives in the tests'
    support package; the trainer runs ``RecordWorkers`` itself."""
    from alignair.train.records import workers
    assert not hasattr(workers, "ParallelRecords") and not hasattr(workers, "parallel_records")


def test_no_error_class_is_never_raised():
    import alignair.errors as errors
    assert not hasattr(errors, "AccessPolicyError")


def test_a_reference_has_one_genotype_format_and_no_unused_locus_helpers():
    """A genotype file is ``genotype.files``' {V|D|J: names}; the reference's own lossy YAML format and its unused
    locus helpers are gone."""
    from alignair.reference import anchors
    from alignair.reference.reference_set import ReferenceSet
    assert not any(hasattr(ReferenceSet, name) for name in ("from_yaml", "to_yaml", "infer_locus", "allele_locus"))
    assert not hasattr(anchors, "gap_from_known")


def test_the_aligner_module_is_the_inference_surface_only():
    """``alignair.aligner`` holds the Aligner and its result; the training names live in ``alignair.train`` (and stay
    on the top-level package)."""
    import alignair
    from alignair import aligner
    assert not {"TrainingConfig", "TrainingRun", "run_training"} & set(aligner.__all__)
    assert not any(hasattr(aligner, name) for name in ("TrainingConfig", "TrainingRun", "run_training"))
    assert {"TrainingConfig", "TrainingRun", "run_training"} <= set(alignair.__all__)


def test_the_registry_resolves_catalog_ids_through_its_sources_directly():
    """No one-line wrapper of ``sources.find_model`` in the cache."""
    from alignair.registry import cache
    assert not hasattr(cache, "_find")


def test_reads_are_tokenized_by_the_one_vocabulary():
    import alignair.data
    assert _gone("alignair.data.tokenizer") and _gone("alignair.data.inference")
    assert not hasattr(alignair.data, "SequenceInputAdapter")
    assert alignair.data.tokenize_batch(["ACGT", "GN"], 6)["tokenized_sequence"].tolist() == [[1, 4, 3, 2, 0, 0],
                                                                                              [3, 5, 0, 0, 0, 0]]


def test_the_boundary_decoder_keeps_no_test_only_helper():
    from alignair.predict import decode
    assert not hasattr(decode, "decoded_boundaries")


def test_a_model_is_identified_by_its_reference_alone():
    import alignair.core as core
    import alignair.model_file as model_file
    from alignair.core.config import AlignAIRConfig
    from alignair.core.setup import ModelSetup
    from alignair.train.config import TrainingConfig
    gone = {"v_allele_count", "j_allele_count", "d_allele_count", "input_space", "label_space"}
    assert not gone & {field.name for field in dataclasses.fields(AlignAIRConfig)}
    assert not gone & {field.name for field in dataclasses.fields(TrainingConfig)}
    assert not any(hasattr(AlignAIRConfig, name) for name in ("allele_count", "from_dataconfigs", "from_reference"))
    assert not any(hasattr(core, name) for name in ("build_model_config", "setup_from_schema"))
    assert not any(hasattr(ModelSetup, name) for name in ("label_schema", "label_counts", "validate"))
    assert not hasattr(model_file, "v3_checkpoint_error")
    assert list(inspect.signature(model_file.save_model).parameters)[:3] == ["path", "model", "reference"]
    assert "label_schema" not in inspect.signature(model_file.save_model).parameters


def test_the_bundle_export_reads_its_provenance_from_the_checkpoint_only():
    from alignair.train import bundle
    assert _gone("alignair.train.build")
    assert "training" not in inspect.signature(bundle.export_bundle).parameters
    assert not hasattr(bundle, "build_dataconfigs") and not hasattr(bundle, "training_plan")


def test_every_renamed_error_class_carries_the_error_suffix():
    """The error classes are named for what failed, with the ``Error`` suffix: the old names are gone."""
    import alignair
    from alignair import io, registry
    from alignair.registry import cache
    from alignair.train import guards
    gone = [(alignair, "DuplicateMetadataId"), (alignair, "PredictionCountMismatch"), (io, "DuplicateMetadataId"),
            (registry, "IntegrityError"), (cache, "IntegrityError"), (guards, "NonFiniteLossError")]
    assert not [name for module, name in gone if hasattr(module, name)]
    assert all(hasattr(module, name) for module, name in (
        (alignair, "DuplicateMetadataIdError"), (alignair, "PredictionCountMismatchError"),
        (registry, "DownloadIntegrityError"), (guards, "NonFiniteTrainingError")))
