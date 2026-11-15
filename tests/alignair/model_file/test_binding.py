"""A model file loads only when its configuration binds its embedded reference, checked before any allocation."""
import json
from dataclasses import replace

import GenAIRR.data as gd
import pytest
import torch

from alignair import model_file
from alignair.core import AlignAIR, prepare_model
from alignair.errors import ConfigurationError, ModelError, ModelIntegrityError
from alignair.model_file import container, serialize
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


@pytest.fixture
def artifact(tmp_path):
    reference = ReferenceSet.from_genotype({'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGTTTT'},
                                            'D': {'D1': 'TAC'}, 'J': {'J1': 'TGGAAA'}})
    reference.gene('V').anchors = {'V2': 3, 'V1': 3}
    reference.gene('J').anchors = {'J1': 0}
    setup = prepare_model(reference, max_seq_length=128, embed_dim=4, trunk_channels=8)
    header = {'reference': {'allele_order_sha256': serialize.allele_order_sha256(reference),
                            'reference_fasta_sha256': serialize.reference_fasta_sha256(reference)}}
    sections = {'reference_json': (serialize.reference_to_json(reference), 'zlib'),
                'config': (serialize.config_to_bytes(setup.config), 'zlib'),
                'weights': (serialize.state_dict_to_bytes(setup.build_model().state_dict()), 'none')}
    return setup, tmp_path / 'model.alignair', header, sections


def test_a_file_whose_configuration_binds_its_reference_loads(artifact):
    setup, path, header, sections = artifact
    container.write_container(path, header, sections)
    loaded = model_file.load_model(path)
    assert loaded.config == setup.config
    assert loaded.reference.gene('V').names == setup.reference.gene('V').names
    for name, weights in serialize.state_dict_from_bytes(sections['weights'][0]).items():
        assert torch.equal(loaded.model.state_dict()[name], weights), name


def test_a_configuration_that_does_not_bind_the_embedded_reference_is_refused_before_allocation(artifact,
                                                                                                monkeypatch):
    setup, path, header, sections = artifact
    sections['config'] = (serialize.config_to_bytes(replace(setup.config, num_chain_types=2)), 'zlib')
    container.write_container(path, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError('model allocated before binding validation')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelIntegrityError, match='chain count') as caught:
        model_file.load_model(path)
    assert isinstance(caught.value.__cause__, ConfigurationError)
    assert caught.value.context['stage'] == 'model_binding'
    assert caught.value.context['path'] == str(path)


@pytest.mark.parametrize('change,detail', [({'vocab_size': 9}, 'vocabulary'),
                                           ({'max_seq_length': 0}, 'max_seq_length')])
def test_an_invalid_stored_configuration_is_the_files_fault(artifact, change, detail):
    setup, path, header, sections = artifact
    stored = {**setup.config.__dict__, **change}
    sections['config'] = (json.dumps(stored).encode(), 'zlib')
    container.write_container(path, header, sections)
    with pytest.raises(ModelIntegrityError, match=detail) as caught:
        model_file.load_model(path)
    assert isinstance(caught.value.__cause__, ConfigurationError)


def test_embedded_reference_contract_error_has_artifact_context(artifact):
    setup, path, header, sections = artifact
    reference = setup.reference
    reference.gene('V').sequences.pop()
    sections['reference_json'] = (serialize.reference_to_json(reference), 'zlib')
    header['reference'] = {'allele_order_sha256': serialize.allele_order_sha256(reference),
                           'reference_fasta_sha256': serialize.reference_fasta_sha256(reference)}
    container.write_container(path, header, sections)
    with pytest.raises(ModelIntegrityError, match='reference allele order') as caught:
        model_file.load_model(path)
    assert caught.value.context['gene'] == 'V'
    assert caught.value.context['path'] == str(path)


def test_a_container_without_reference_json_is_refused_as_written_before_alignair_4(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                          max_seq_length=128, embed_dim=4, trunk_channels=8)
    path = tmp_path / 'legacy.alignair'
    model_file.save_model(path, setup.build_model(), setup.reference, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    header = container.read_header(path)
    header.pop('_sections_base', None)
    sections = {name: (container.read_section(path, name), info['codec'])
                for name, info in header['sections'].items() if name != 'reference_json'}
    container.write_container(path, header, sections)
    with pytest.raises(ModelError, match='no reference_json: it was written before AlignAIR 4') as caught:
        model_file.load_model(path)
    assert caught.value.context['path'] == str(path) and 'alignair<4' in caught.value.hint


def test_save_rejects_invalid_binding_before_touching_destination(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                          max_seq_length=128, embed_dim=4, trunk_channels=8)
    model = AlignAIR(replace(setup.config, num_chain_types=2))
    path = tmp_path / 'existing.alignair'
    path.write_bytes(b'keep existing artifact')
    with pytest.raises(ConfigurationError) as caught:
        model_file.save_model(path, model, setup.reference, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    assert caught.value.context['stage'] == 'model_binding'
    assert path.read_bytes() == b'keep existing artifact'
