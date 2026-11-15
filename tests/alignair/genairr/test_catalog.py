"""One adapter resolves, lists and names GenAIRR's dataconfigs, whatever form a caller gives them in."""
import copy
import os
import subprocess
import sys

import pytest

import GenAIRR.data as gd
from alignair.errors import ConfigurationError
from alignair.genairr import name_of, resolve_dataconfig

_V, _D, _J = (f"examples/custom_reference/{g}.fasta" for g in "vdj")


@pytest.mark.component
def test_a_builtin_name_resolves_to_its_dataconfig():
    assert resolve_dataconfig("HUMAN_IGH_OGRDB") is gd.HUMAN_IGH_OGRDB
    assert resolve_dataconfig("human_igh_ogrdb") is gd.HUMAN_IGH_OGRDB
    assert resolve_dataconfig(gd.HUMAN_IGK_OGRDB) is gd.HUMAN_IGK_OGRDB


@pytest.mark.integration
def test_a_name_outside_the_top_level_shortcuts_resolves():
    from alignair.train.records.streams import CurriculumStreamPolicy, training_records
    assert "mouse" in str(resolve_dataconfig("MOUSE_IGH_IMGT").metadata.species).lower()
    record = next(iter(training_records(("MOUSE_IGH_IMGT",), policy=CurriculumStreamPolicy(mode="amplicon"), seed=1)))
    assert record["sequence"]


@pytest.mark.component
@pytest.mark.parametrize("spec", ["HUMAN_IGH_NOPE", 3, None])
def test_an_unknown_name_is_refused_with_the_list_hint(spec):
    with pytest.raises(ConfigurationError) as caught:
        resolve_dataconfig(spec)
    assert "alignair reference list" in caught.value.hint


@pytest.mark.component
@pytest.mark.skipif(not all(os.path.exists(p) for p in (_V, _D, _J)), reason="example FASTAs absent")
def test_name_of_names_builtins_and_their_copies_but_not_custom_ones():
    from alignair.genairr.custom import build_dataconfigs
    assert name_of(gd.HUMAN_TCRB_IMGT) == "HUMAN_TCRB_IMGT"
    assert name_of(copy.deepcopy(gd.HUMAN_IGH_OGRDB)) == "HUMAN_IGH_OGRDB"
    (custom,), _ = build_dataconfigs(v_fasta=_V, j_fasta=_J, d_fasta=_D, chain_type="BCR_HEAVY")
    assert name_of(custom) is None


@pytest.mark.integration
def test_model_file_imports_no_simulator():
    found = subprocess.run([sys.executable, "-c", "import sys, alignair.model_file; print('GenAIRR' in sys.modules)"],
                           capture_output=True, text=True, env=dict(os.environ), check=True)
    assert found.stdout.strip() == "False"
