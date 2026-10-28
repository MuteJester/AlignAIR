"""Public loading rejects overrides that change allele biology or locus interpretation."""
from copy import deepcopy

import GenAIRR.data as gd
import pytest

from alignair import load_model, model_file
from alignair.core import prepare_model
from alignair.errors import DataContractError, ModelError
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


@pytest.fixture(scope='module')
def artifact(tmp_path_factory):
    configs = [gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB]
    reference = ReferenceSet.from_dataconfigs(*configs)
    setup = prepare_model(reference, max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    path = tmp_path_factory.mktemp('reference-identity') / 'model.alignair'
    model_file.save_model(path, setup.build_model(), dataconfigs=configs, training={},
                          include_trusted_pickle=False)
    return path, reference


def test_matching_override_is_returned_with_the_model(artifact):
    path, expected = artifact
    candidate = deepcopy(expected)
    model, reference = load_model(path, reference=candidate)
    assert reference is candidate and model.cfg.num_chain_types == 2
    assert reference.locus_names() == ('IGK', 'IGL')


@pytest.mark.parametrize('field', ['locus_order', 'locus_ranges', 'chain_type', 'sequences', 'anchors', 'gapped'])
def test_same_sized_override_cannot_change_reference_meaning(artifact, field):
    path, expected = artifact
    candidate = deepcopy(expected)
    if field == 'locus_order':
        candidate.loci.reverse()
    elif field == 'locus_ranges':
        start, end = candidate.loci[0].ranges['V']
        candidate.loci[0].ranges['V'] = (start, end - 1)
    elif field == 'chain_type':
        candidate.loci[0].chain_type = 'different'
    elif field == 'sequences':
        sequence = candidate.gene('V').sequences[0]
        candidate.gene('V').sequences[0] = ('A' if sequence[0] != 'A' else 'T') + sequence[1:]
    elif field == 'anchors':
        name = next(iter(candidate.gene('V').anchors))
        candidate.gene('V').anchors[name] += 1
    else:
        name = next(iter(candidate.gene('V').gapped))
        candidate.gene('V').gapped[name] += '.'
    with pytest.raises(ModelError) as caught:
        load_model(path, reference=candidate)
    assert caught.value.context['field'] == ('loci' if field.startswith('locus_') or field == 'chain_type' else field)
    assert caught.value.context['stage'] == 'reference_identity'
    assert caught.value.context['path'] == str(path)


def test_override_cannot_supply_an_inconsistent_gene_index(artifact):
    path, expected = artifact
    candidate = deepcopy(expected)
    candidate.gene('V').index[candidate.gene('V').names[0]] = 9
    with pytest.raises(DataContractError) as caught:
        load_model(path, reference=candidate)
    assert caught.value.context['stage'] == 'reference_identity'
    assert caught.value.context['path'] == str(path)
