"""``<out>.run.json`` records what the calls came from: the ``--reference`` and ``--genotype`` given, the fingerprints
of the reference actually bound (not the model's embedded one when another was bound), the embedded one's beside them,
and the genotype bank's allele order."""
import hashlib
import json

import pytest

from alignair.cli.main import main
from alignair.genotype import genotype_bank
from alignair.model_file import load_model, read_metadata, serialize
from alignair.reference import ReferenceSet, annotate
from tests.alignair.predict.golden_rows import TEST_CHIP

pytestmark = pytest.mark.integration


def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def _run(tmp_path, *extra):
    ref = load_model(str(TEST_CHIP)).reference
    read = ref.gene("V").sequences[5] + "GGGAAA" + ref.gene("D").sequences[2] + "CCT" + ref.gene("J").sequences[3]
    (tmp_path / "r.fasta").write_text(f">r\n{read}\n")
    out = tmp_path / "o.tsv"
    assert main(["predict", "--model", str(TEST_CHIP), "--input", str(tmp_path / "r.fasta"), "--out", str(out),
                 "--device", "cpu", "--offline", "--quiet", *extra]) == 0
    return json.loads((tmp_path / "o.tsv.run.json").read_text()), ref


def test_a_plain_run_records_the_embedded_reference(tmp_path):
    run, ref = _run(tmp_path)
    card = read_metadata(str(TEST_CHIP))["reference"]
    assert run["reference_fasta_sha256"] == card["reference_fasta_sha256"]
    assert run["allele_order_sha256"] == card["allele_order_sha256"]
    assert run["reference"]["source"] == "embedded" and run["reference"]["path"] is None
    assert run["genotype"] is None
    assert run["command"]["reference"] is None and run["command"]["genotype"] is None


def test_a_bound_reference_and_a_genotype_are_recorded_with_the_fingerprints_of_what_was_bound(tmp_path):
    ref = load_model(str(TEST_CHIP)).reference
    v = ref.gene("V")
    novel = v.sequences[0][:100] + ("A" if v.sequences[0][100] != "A" else "C") + v.sequences[0][101:]
    fasta = tmp_path / "germline.fasta"
    fasta.write_text(serialize.reference_fasta(ref) + f">{v.names[0].split('*')[0]}*99\n{novel}\n")
    genotype = tmp_path / "donor.yaml"
    genotype.write_text(f"V: [{v.names[5]}, {v.names[0].split('*')[0]}*99]\n")
    run, _ = _run(tmp_path, "--reference", str(fasta), "--genotype", str(genotype))
    bound = annotate(ReferenceSet.from_fasta(str(fasta)), known=ref)
    card = read_metadata(str(TEST_CHIP))["reference"]
    assert run["command"]["reference"] == str(fasta) and run["command"]["genotype"] == str(genotype)
    assert run["reference"] == {"source": "--reference", "path": str(fasta), "file_sha256": _sha(fasta),
                                "embedded_reference_fasta_sha256": card["reference_fasta_sha256"],
                                "embedded_allele_order_sha256": card["allele_order_sha256"]}
    assert run["reference_fasta_sha256"] == serialize.reference_fasta_sha256(bound) != card["reference_fasta_sha256"]
    assert run["allele_order_sha256"] == serialize.allele_order_sha256(bound)
    bank = genotype_bank(bound, {"V": [v.names[5], f"{v.names[0].split('*')[0]}*99"]})
    assert run["genotype"] == {"path": str(genotype), "file_sha256": _sha(genotype),
                               "allele_order_sha256": serialize.allele_order_sha256(bank)}


def test_the_run_records_the_alignair_that_predicted_and_the_one_that_made_the_model(tmp_path):
    import alignair
    run, _ = _run(tmp_path)
    assert run["alignair_version"] == alignair.__version__
    assert run["model_created_by_alignair"] == read_metadata(str(TEST_CHIP))["created_by_alignair"]
