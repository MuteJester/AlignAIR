"""End-to-end workflow tests on the real example fixtures under ``examples/``.

These prove that 10x Genomics (Cell Ranger) and bulk-AIRR data survive a full
``alignair predict`` run: the per-cell / per-sample metadata is carried into the
output, the output is valid AIRR-C, and downstream tooling (the ``airr`` library,
Scirpy-style group-by-cell) can read it back. The example files ARE the fixtures,
so the documented workflow is exactly what is tested."""
import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("GenAIRR")

from alignair import cli
from alignair.serialization.dnalignair_bundle import save_dnalignair_bundle
from alignair.config.dnalignair_config import DNAlignAIRConfig
from alignair.core.dnalignair import DNAlignAIR

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"


@pytest.fixture(scope="module")
def igh_bundle(tmp_path_factory):
    """A tiny (untrained) IGH bundle — enough to exercise the I/O + metadata path."""
    torch.manual_seed(0)
    model = DNAlignAIR(DNAlignAIRConfig(d_model=32, n_layers=1, nhead=2, dim_feedforward=64))
    d = tmp_path_factory.mktemp("bundle")
    save_dnalignair_bundle(d, model=model, dataconfigs=["HUMAN_IGH_OGRDB"], locus="IGH")
    return str(d)


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def test_tenx_workflow_preserves_cell_metadata(igh_bundle, tmp_path):
    fasta = EXAMPLES / "10x" / "filtered_contig.fasta"
    annots = EXAMPLES / "10x" / "filtered_contig_annotations.csv"
    out = tmp_path / "tenx_airr.tsv"
    cli.main(["predict", str(fasta), "-o", str(out), "--model", igh_bundle,
              "--metadata", str(annots), "--device", "cpu", "--quiet"])

    rows = _rows(out)
    by_id = {r["sequence_id"]: r for r in rows}
    # every contig aligned, keyed by its 10x contig_id
    assert set(by_id) == {
        "AAACCTGAGAAACCAT-1_contig_1", "AAACCTGAGAAACCAT-1_contig_2",
        "AAACCTGAGAAGGCCT-1_contig_1", "AAACCTGCATGGTAGG-1_contig_1",
    }
    # 10x annotations carried through verbatim, joined by contig_id
    src = {r["contig_id"]: r for r in csv.DictReader(open(annots))}
    for cid, row in by_id.items():
        assert row["barcode"] == src[cid]["barcode"]
        assert row["umis"] == src[cid]["umis"]
        assert row["chain"] == src[cid]["chain"]
        assert row["raw_clonotype_id"] == src[cid]["raw_clonotype_id"]
        assert row["v_call"]                       # AlignAIR still produced a call


def test_tenx_output_is_airr_c_valid(igh_bundle, tmp_path):
    airr = pytest.importorskip("airr")
    out = tmp_path / "tenx_airr.tsv"
    cli.main(["predict", str(EXAMPLES / "10x" / "filtered_contig.fasta"), "-o", str(out),
              "--model", igh_bundle, "--metadata",
              str(EXAMPLES / "10x" / "filtered_contig_annotations.csv"),
              "--device", "cpu", "--quiet"])
    assert airr.validate_rearrangement(str(out))


def test_tenx_readback_groups_by_cell(igh_bundle, tmp_path):
    """Downstream (Scirpy/Change-O) reconstruct cells by grouping rows on barcode."""
    out = tmp_path / "tenx_airr.tsv"
    cli.main(["predict", str(EXAMPLES / "10x" / "filtered_contig.fasta"), "-o", str(out),
              "--model", igh_bundle, "--metadata",
              str(EXAMPLES / "10x" / "filtered_contig_annotations.csv"),
              "--device", "cpu", "--quiet"])
    cells = {}
    for r in _rows(out):
        cells.setdefault(r["barcode"], []).append(r)
    assert set(cells) == {"AAACCTGAGAAACCAT-1", "AAACCTGAGAAGGCCT-1", "AAACCTGCATGGTAGG-1"}
    assert len(cells["AAACCTGAGAAACCAT-1"]) == 2          # this cell has two contigs
    assert all(r["v_call"] for rs in cells.values() for r in rs)


def test_airr_library_reads_output_with_extra_columns(igh_bundle, tmp_path):
    """The Immcantation `airr` parser round-trips the output, extra columns intact."""
    airr = pytest.importorskip("airr")
    out = tmp_path / "tenx_airr.tsv"
    cli.main(["predict", str(EXAMPLES / "10x" / "filtered_contig.fasta"), "-o", str(out),
              "--model", igh_bundle, "--metadata",
              str(EXAMPLES / "10x" / "filtered_contig_annotations.csv"),
              "--device", "cpu", "--quiet"])
    recs = list(airr.read_rearrangement(str(out)))
    assert len(recs) == 4
    assert {r["barcode"] for r in recs} >= {"AAACCTGAGAAACCAT-1"}
    assert all(r.get("umis") for r in recs)


def test_bulk_airr_input_joins_sample_metadata(igh_bundle, tmp_path):
    reads = EXAMPLES / "airr" / "reads.tsv"
    meta = EXAMPLES / "airr" / "sample_metadata.tsv"
    out = tmp_path / "out.tsv"
    cli.main(["predict", str(reads), "-o", str(out), "--model", igh_bundle,
              "--metadata", str(meta), "--keep-columns", "sample_id,subject_id,tissue,timepoint",
              "--device", "cpu", "--quiet"])
    rows = _rows(out)
    by_id = {r["sequence_id"]: r for r in rows}
    assert set(by_id) == {"M1S1-00001", "M1S1-00002", "M2S1-00001"}
    assert by_id["M1S1-00001"]["sample_id"] == "M1S1"
    assert by_id["M1S1-00001"]["subject_id"] == "donor-M1"
    assert by_id["M2S1-00001"]["tissue"] == "bone_marrow"
    assert by_id["M2S1-00001"]["timepoint"] == "day14"
