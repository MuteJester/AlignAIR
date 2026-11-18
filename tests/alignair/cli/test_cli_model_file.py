import pytest

from alignair.core import prepare_model
from alignair.reference.reference_set import ReferenceSet
import GenAIRR.data as gd
from alignair.core import AlignAIR
from alignair import model_file as mf
from alignair.cli.main import main

pytestmark = pytest.mark.integration


def _save(tmp_path):
    cfg = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB), max_seq_length=576).config
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), AlignAIR(cfg), ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                  dataconfigs=["HUMAN_IGH_OGRDB"],
                  training={"steps": 2, "batch_size": 3})
    return str(p)


def test_cli_info(tmp_path, capsys):
    p = _save(tmp_path)
    assert main(["info", p]) == 0
    out = capsys.readouterr().out
    assert "AlignAIR" in out and "HUMAN_IGH_OGRDB" in out and "total_sequences_seen" in out


def test_info_prints_the_allele_counts_and_the_allele_order(tmp_path, capsys):
    p = _save(tmp_path)
    assert main(["info", p]) == 0
    card = capsys.readouterr().out
    assert "alleles:   V=198 D=33 J=7" in card and "allele order:" in card


@pytest.mark.parametrize("dataconfig,has_d,expected", [
    (gd.HUMAN_IGK_OGRDB, None, "D=0"),        # a light-chain model: its reference has no D
    (gd.HUMAN_IGH_OGRDB, False, "D=0"),       # a model built without a D on a reference that has one
])
def test_the_card_counts_the_alleles_the_model_calls(tmp_path, capsys, dataconfig, has_d, expected):
    """The card's allele counts are the model's genes' (a model without a D calls no D allele), not the embedded
    reference's genes."""
    reference = ReferenceSet.from_dataconfigs(dataconfig)
    setup = prepare_model(reference, has_d=has_d, max_seq_length=128, embed_dim=4, trunk_channels=8)
    p = tmp_path / "m.alignair"
    mf.save_model(str(p), setup.build_model(), setup.reference, training={"steps": 1, "batch_size": 1})
    assert main(["info", str(p)]) == 0
    assert f"V={len(reference.gene('V'))} {expected} J={len(reference.gene('J'))}" in capsys.readouterr().out


def test_info_prints_the_card_of_an_alignair_3_model_file(tmp_path, capsys):
    """An AlignAIR 3 model file (no ``reference_json``; its header stores the allele counts) still gets its card, as
    before; only loading it is refused."""
    from alignair.model_file import container
    p = _save(tmp_path)
    header = container.read_header(p)
    header.pop("_sections_base", None)
    sections = {name: (container.read_section(p, name), info["codec"])
                for name, info in header["sections"].items() if name != "reference_json"}
    header["model"]["allele_counts"] = {"v": 3, "d": 2, "j": 1}
    container.write_container(p, header, sections)
    assert main(["info", p]) == 0
    assert "alleles:   V=3 D=2 J=1" in capsys.readouterr().out


def test_cli_reference_export_writes_the_fasta(tmp_path, capsys):
    p = _save(tmp_path)
    fasta = tmp_path / "ref.fasta"
    assert main(["reference", "export", p, "--fasta", str(fasta)]) == 0
    assert fasta.read_text().startswith(">")


def test_reference_export_never_unpickles_a_dataconfig_section(tmp_path, capsys):
    """A model file's pickled sections are never loaded by the CLI: the export takes no --dataconfig."""
    p = _save(tmp_path)
    out = tmp_path / "dc.pkl"
    with pytest.raises(SystemExit) as caught:
        main(["reference", "export", p, "--dataconfig", str(out)])
    assert caught.value.code == 2 and not out.exists()


def test_reference_export_needs_somewhere_to_write(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["reference", "export", _save(tmp_path)])
    assert caught.value.code == 2


def test_the_duplicate_export_reference_verb_is_gone(tmp_path, capsys):
    with pytest.raises(SystemExit) as caught:
        main(["export-reference", _save(tmp_path), "--fasta", str(tmp_path / "ref.fasta")])
    assert caught.value.code == 2


def _fasta_records(text: str) -> dict[str, str]:
    records, name = {}, None
    for line in text.splitlines():
        if line.startswith(">"):
            name = line[1:].split()[0]
            records[name] = ""
        elif name:
            records[name] += line.strip()
    return records


def test_reference_export_gapped_writes_imgt_gapped_v_for_change_o(tmp_path, capsys):
    """``--gapped`` writes each V allele IMGT-gapped (as Change-O's CreateGermlines.py needs) and D and J as they
    are."""
    from tests.alignair.predict.golden_rows import TEST_CHIP
    plain, gapped = tmp_path / "plain.fasta", tmp_path / "gapped.fasta"
    assert main(["reference", "export", str(TEST_CHIP), "--fasta", str(plain)]) == 0
    assert main(["reference", "export", str(TEST_CHIP), "--fasta", str(gapped), "--gapped"]) == 0
    reference = mf.load_model(str(TEST_CHIP)).reference
    plain_records, gapped_records = _fasta_records(plain.read_text()), _fasta_records(gapped.read_text())
    assert list(gapped_records) == list(plain_records)
    v = reference.gene("V")
    assert all(gapped_records[name] == v.gapped[name] for name in v.names)
    assert all("." in gapped_records[name] for name in v.names)
    assert all(gapped_records[name] == plain_records[name] for name in plain_records if name not in v.names)


def test_reference_export_without_gapped_is_unchanged(tmp_path, capsys):
    from tests.alignair.predict.golden_rows import TEST_CHIP
    fasta = tmp_path / "ref.fasta"
    assert main(["reference", "export", str(TEST_CHIP), "--fasta", str(fasta)]) == 0
    assert fasta.read_text() == mf.read_reference(TEST_CHIP)


def test_a_gapped_export_of_a_reference_without_imgt_gaps_is_refused(tmp_path, capsys):
    """A V allele with no IMGT-gapped form is named, not written ungapped among gapped ones."""
    import copy
    import dataclasses
    reference = copy.deepcopy(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB))
    v = reference.gene("V")
    first = v.names[0]
    reference.genes["V"] = dataclasses.replace(v, gapped={n: g for n, g in v.gapped.items() if n != first})
    cfg = prepare_model(reference, max_seq_length=576).config
    path = tmp_path / "m.alignair"
    mf.save_model(str(path), AlignAIR(cfg), reference, training={"steps": 2, "batch_size": 3})
    assert main(["--error-format", "json", "reference", "export", str(path), "--fasta",
                 str(tmp_path / "g.fasta"), "--gapped"]) != 0
    import json
    payload = json.loads(capsys.readouterr().err)
    assert payload["code"] == "ALIGNAIR_CONFIG_INVALID" and first in payload["context"]["alleles"]
