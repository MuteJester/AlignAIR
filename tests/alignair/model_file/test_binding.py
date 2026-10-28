"""Schema-less artifact compatibility must not bypass reference/head binding preflight."""
from dataclasses import replace

import GenAIRR.data as gd
import pytest
import torch

from alignair import model_file
from alignair.core import AlignAIR, prepare_model
from alignair.errors import ConfigurationError, ModelIntegrityError
from alignair.model_file import container, serialize
from alignair.reference import ReferenceSet

pytestmark = pytest.mark.integration


@pytest.fixture(params=['nucleotide', 'protein'])
def artifact(request, tmp_path):
    reference = ReferenceSet.from_genotype({'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGTTTT'},
                                            'D': {'D1': 'TAC'}, 'J': {'J1': 'TGGAAA'}})
    reference.gene('V').anchors = {'V2': 3, 'V1': 3}
    reference.gene('J').anchors = {'J1': 0}
    setup = prepare_model(reference, input_space=request.param, max_seq_length=128,
                          embed_dim=4, filters=4, block_out=8)
    header = {'reference': {'allele_order_sha256': serialize.allele_order_sha256(reference),
                            'reference_fasta_sha256': serialize.reference_fasta_sha256(reference)}}
    sections = {'reference_json': (serialize.reference_to_json(reference), 'zlib'),
                'config': (serialize.config_to_bytes(setup.config), 'zlib'),
                'weights': (serialize.state_dict_to_bytes(setup.build_model().state_dict()), 'none')}
    return setup, tmp_path / 'model.alignair', header, sections


def test_older_allele_artifact_loads_without_schema_when_head_dimensions_match(artifact):
    setup, path, header, sections = artifact
    container.write_container(path, header, sections)
    loaded = model_file.load_model(path)
    assert loaded.config == setup.config
    assert loaded.reference.gene('V').names == setup.reference.gene('V').names
    for name, weights in serialize.state_dict_from_bytes(sections['weights'][0]).items():
        assert torch.equal(loaded.model.state_dict()[name], weights), name


@pytest.mark.parametrize('change,detail', [
    ({'v_allele_count': 5}, 'head size'), ({'d_allele_count': 5}, 'head size'),
    ({'j_allele_count': 5}, 'head size'), ({'has_d': False}, 'disabled D'),
    ({'num_chain_types': 2}, 'chain count'), ({'vocab_size': 9}, 'vocabulary'),
    ({'max_seq_length': 0}, 'max_seq_length'),
])
def test_missing_schema_never_bypasses_binding_checks_before_allocation(artifact, change, detail, monkeypatch):
    setup, path, header, sections = artifact
    sections['config'] = (serialize.config_to_bytes(replace(setup.config, **change)), 'zlib')
    container.write_container(path, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError('model allocated before binding validation')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelIntegrityError, match=detail) as caught:
        model_file.load_model(path)
    assert isinstance(caught.value.__cause__, ConfigurationError)
    assert caught.value.context['stage'] == 'model_binding'
    assert caught.value.context['path'] == str(path)


def test_protein_class_artifact_requires_schema_even_when_widths_equal_allele_counts(artifact, monkeypatch):
    setup, path, header, sections = artifact
    cfg = replace(setup.config, label_space='protein_classes', classification_mode='categorical')
    sections['config'] = (serialize.config_to_bytes(cfg), 'zlib')
    container.write_container(path, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError('class IDs inferred without stored labels')
    monkeypatch.setattr(AlignAIR, '__init__', forbidden)
    with pytest.raises(ModelIntegrityError, match='ordered class labels') as caught:
        model_file.load_model(path)
    assert caught.value.context['field'] == 'label_schema'
    assert caught.value.hint == 'Supply the saved label schema or protein reference; do not infer class IDs.'


def test_class_artifact_preserves_saved_order_without_retranslating_reference(artifact, monkeypatch):
    source, path, header, sections = artifact
    setup = prepare_model(source.reference, input_space=source.input_space, label_space='protein_classes',
                          max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    schema = setup.label_schema()
    schema['genes']['v'].reverse()
    header['model'] = {'label_schema': schema}
    sections['config'] = (serialize.config_to_bytes(setup.config), 'zlib')
    sections['weights'] = (serialize.state_dict_to_bytes(setup.build_model().state_dict()), 'none')
    container.write_container(path, header, sections)
    def forbidden(*args, **kwargs):
        raise AssertionError('saved class order was reconstructed')
    monkeypatch.setattr('alignair.core.setup.build_protein_reference', forbidden)
    loaded = model_file.load_model(path)
    assert loaded.metadata['model']['label_schema'] == schema


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


def test_trusted_legacy_container_still_verifies_available_reference_hashes(tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                          max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    path = tmp_path / 'legacy.alignair'
    model_file.save_model(path, setup.build_model(), dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    header = container.read_header(path)
    header.pop('_sections_base', None)
    sections = {name: (container.read_section(path, name), info['codec'])
                for name, info in header['sections'].items() if name != 'reference_json'}
    header['reference']['allele_order_sha256'] = '0' * 64
    container.write_container(path, header, sections)
    with pytest.raises(ModelIntegrityError, match='allele_order_sha256') as caught:
        model_file.load_model(path, trust_pickle=True)
    assert caught.value.context['stage'] == 'model_binding'


@pytest.mark.parametrize('problem', ['head_count', 'missing_class_schema'])
def test_save_rejects_invalid_binding_before_touching_destination(problem, tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                          max_seq_length=128, embed_dim=4, filters=4, block_out=8)
    cfg = (replace(setup.config, v_allele_count=1) if problem == 'head_count'
           else replace(setup.config, label_space='protein_classes'))
    model = AlignAIR(cfg)
    path = tmp_path / 'existing.alignair'
    path.write_bytes(b'keep existing artifact')
    with pytest.raises(ConfigurationError) as caught:
        model_file.save_model(path, model, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    assert caught.value.context['stage'] == 'model_binding'
    assert path.read_bytes() == b'keep existing artifact'
