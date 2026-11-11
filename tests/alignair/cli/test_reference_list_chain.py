"""``alignair reference list --chain`` matches whole tokens of a reference's chain type, name and locus, not
substrings: ``IGH`` no longer matches BCR_LIGHT (whose "LIGHT" holds "IGH"), and the documented family filter ``TCR``
still lists every TCR reference."""
import json

import pytest

from alignair.cli.main import main
from alignair.genairr import list_dataconfigs

pytestmark = pytest.mark.component


def _listed(capsys, *args):
    assert main(["reference", "list", "--json", *args]) == 0
    return json.loads(capsys.readouterr().out)


def test_igh_lists_only_heavy_chain_references(capsys):
    rows = _listed(capsys, "--chain", "IGH")
    assert rows and {r["locus"] for r in rows} == {"IGH"}
    assert len(rows) == sum(r["locus"] == "IGH" for r in list_dataconfigs())


def test_tcr_lists_every_tcr_reference_and_nothing_else(capsys):
    rows = _listed(capsys, "--chain", "TCR")
    assert len(rows) == sum(r["chain_type"].startswith("TCR_") for r in list_dataconfigs())
    assert all(r["chain_type"].startswith("TCR_") for r in rows)


@pytest.mark.parametrize("query, chains", [
    ("BCR_LIGHT_KAPPA", {"BCR_LIGHT_KAPPA"}), ("light", {"BCR_LIGHT_KAPPA", "BCR_LIGHT_LAMBDA"}),
    ("trb", {"TCR_BETA"}), ("IGK", {"BCR_LIGHT_KAPPA"})])
def test_a_chain_type_a_token_of_it_or_a_locus_selects_its_references(capsys, query, chains):
    rows = _listed(capsys, "--chain", query)
    assert rows and {r["chain_type"] for r in rows} == chains


def test_species_and_chain_combine(capsys):
    rows = _listed(capsys, "--species", "Human", "--chain", "IGH")
    assert {r["name"] for r in rows} == {"HUMAN_IGH_EXTENDED", "HUMAN_IGH_IMGT", "HUMAN_IGH_OGRDB"}
