"""The chip (chip-v1) end to end on the human IGH reference: it trains from GenAIRR, reloads through a model file bound to its
reference, predicts AIRR rows and resumes."""
import numpy as np
import pytest

from alignair import TrainingConfig, run_training
from alignair.model_file import read_training_state


pytestmark = pytest.mark.integration

OPTIONS = {"architecture": "chip-v1", "embed_dim": 8}


@pytest.mark.parametrize("channels", [64, 16])
@pytest.mark.heavy
def test_a_tiny_chip_trains_reloads_predicts_airr_rows_and_resumes(tmp_path, channels):
    request = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", steps=2, batch_size=2, device="cpu", seed=31,
                                          val_every=1, model_options={**OPTIONS, "trunk_channels": channels})
    losses, metrics = [], []
    run = run_training(request, output_dir=tmp_path / "first",
                       on_iteration=lambda item: losses.append(item.result.total))
    assert len(losses) == 2 and np.isfinite(losses).all()
    aligner = run.best_aligner(device="cpu")
    assert aligner.model.cfg.architecture == "chip-v1" and aligner.model.core.bank is not None
    # the trunk's width, as the model file rebuilds it
    assert aligner.model.core.trunk.project.out_channels == channels
    sequences = ["CAGGTGCAGCTGGTGCAGTCTGGGGCTGAGGTGAAGAAGCCTGGGGCCTCAGTGAAGGTCTCCTGCAAGGCTTCTGGATACACCTTC",
                 "ACGTN"]
    result = aligner.predict(sequences, batch_size=2)
    rows = list(result)
    # the V read keeps its V call; the 5 nt read holds no V (predict.v_presence)
    assert len(rows) == 2 and rows[0]["v_call"] and (rows[1]["v_call"], rows[1]["v_presence"]) == ("", "absent")
    raw = aligner.predict_raw(sequences, batch_size=1)
    assert raw["v_allele"].shape == (2, len(aligner.reference.gene("V")))
    assert not any(key.startswith("chip_") for key in raw)
    steps = []
    resumed = run_training(request.with_(steps=3, resume_path=run.model_path), output_dir=tmp_path / "resumed",
                           on_iteration=lambda item: steps.append(item.step))
    assert steps == [3] and read_training_state(resumed.model_path).step == 3


@pytest.mark.heavy
def test_predict_with_another_reference_calls_only_its_alleles(tmp_path):
    """``alignair predict --reference``: the chip (reference-agnostic) calls from the FASTA it is given, a novel V the
    training reference lacks included."""
    import csv

    import GenAIRR.data as gd

    from alignair.cli.main import build_parser
    from alignair.reference.reference_set import ReferenceSet

    request = TrainingConfig.from_genairr("HUMAN_IGH_OGRDB", steps=2, batch_size=2, device="cpu", seed=31,
                                          model_options=OPTIONS)
    run = run_training(request, output_dir=tmp_path / "train")
    reference = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)
    v = reference.gene("V")
    novel = v.sequences[0][:100] + ("A" if v.sequences[0][100] != "A" else "C") + v.sequences[0][101:]
    alleles = {**dict(zip(v.names[:20], v.sequences[:20])), "IGHV-novel*01": novel}
    for gene in ("D", "J"):
        alleles.update(zip(reference.gene(gene).names, reference.gene(gene).sequences))
    fasta = tmp_path / "germline.fasta"
    fasta.write_text("".join(f">{name}\n{sequence}\n" for name, sequence in alleles.items()))
    reads = tmp_path / "reads.fasta"
    reads.write_text(f">r0\n{novel}GGGACCCTGGTCACCGTCTCCTCAG\n>r1\n{v.sequences[3]}\n")
    out = tmp_path / "out.tsv"
    args = build_parser().parse_args(["predict", "--model", str(run.model_path), "--reference", str(fasta),
                                      "--input", str(reads), "--out", str(out), "--device", "cpu",
                                      "--no-run-metadata", "--quiet", "--max-assembly-failures", "1"])
    assert args.func(args) == 0
    with open(out) as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    called = {name for row in rows for name in row["v_call"].split(",") if name}
    assert len(rows) == 2 and called and called <= set(v.names[:20]) | {"IGHV-novel*01"}
