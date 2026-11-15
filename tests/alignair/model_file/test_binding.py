"""Schema-less artifact compatibility must not bypass reference/head binding preflight."""
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


@pytest.fixture(params=['nucleotide'])
def artifact(request, tmp_path):
    reference = ReferenceSet.from_genotype({'V': {'V2': 'ATGTGTAAA', 'V1': 'ATGTGTTTT'},
                                            'D': {'D1': 'TAC'}, 'J': {'J1': 'TGGAAA'}})
    reference.gene('V').anchors = {'V2': 3, 'V1': 3}
    reference.gene('J').anchors = {'J1': 0}
    setup = prepare_model(reference, input_space=request.param, max_seq_length=128,
                          embed_dim=4, trunk_channels=8)
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
    model_file.save_model(path, setup.build_model(), dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    header = container.read_header(path)
    header.pop('_sections_base', None)
    sections = {name: (container.read_section(path, name), info['codec'])
                for name, info in header['sections'].items() if name != 'reference_json'}
    container.write_container(path, header, sections)
    with pytest.raises(ModelError, match='no reference_json: it was written before AlignAIR 4') as caught:
        model_file.load_model(path)
    assert caught.value.context['path'] == str(path) and 'alignair<4' in caught.value.hint


@pytest.mark.parametrize('problem', ['head_count'])
def test_save_rejects_invalid_binding_before_touching_destination(problem, tmp_path):
    setup = prepare_model(ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB),
                          max_seq_length=128, embed_dim=4, trunk_channels=8)
    cfg = replace(setup.config, v_allele_count=1)
    model = AlignAIR(cfg)
    path = tmp_path / 'existing.alignair'
    path.write_bytes(b'keep existing artifact')
    with pytest.raises(ConfigurationError) as caught:
        model_file.save_model(path, model, dataconfigs=[gd.HUMAN_IGH_OGRDB], training={})
    assert caught.value.context['stage'] == 'model_binding'
    assert path.read_bytes() == b'keep existing artifact'
