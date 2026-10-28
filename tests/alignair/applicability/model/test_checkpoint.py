import pytest

from alignair.applicability.checkpoint import build_provenance, verify_provenance, load_checkpoint, save_checkpoint
from alignair.applicability.fusion import fit_fusion
from alignair.errors import ConfigurationError, ModelFormatError, ModelIntegrityError, OperationIOError, ArtifactIntegrityError

pytestmark = pytest.mark.component


def _prov():
    return build_provenance(bank_checksum="BC", training_manifest_hash="TM", partition_hash="PH",
                            seed=7, loss_mask_config={"oe_cap": 0.15})


def test_provenance_binds_and_verifies():
    p = _prov()
    d = p.to_dict()
    assert len(p.digest()) == 64
    for k in ("tokenizer", "bank_checksum", "feature_order", "class_map_hash", "partition_hash",
              "seed", "software_versions", "training_manifest_hash", "loss_mask_config"):
        assert k in d
    verify_provenance(d, p)                          # identical -> ok


def test_bank_checksum_or_feature_order_change_hard_fails_load():
    p = _prov()
    with pytest.raises(ValueError):
        verify_provenance(dict(p.to_dict(), bank_checksum="OTHER"), p)
    with pytest.raises(ValueError):
        verify_provenance(dict(p.to_dict(), feature_order=["shuffled"]), p)


def test_train_only_normalization_differs_from_full_cohort():
    # Safeguard 6: stats fit on TRAIN partition only must differ from a whole-cohort fit
    ft = fit_fusion(["a", "b"], [[0, 0], [2, 2]], "BC")
    ff = fit_fusion(["a", "b"], [[0, 0], [2, 2], [10, 10]], "BC")
    assert not (ft.mean == ff.mean).all()


@pytest.mark.parametrize('field,value', [('tokenizer', 'other'), ('bank_checksum', 'other'),
    ('feature_order', ['other']), ('class_map_hash', 'other'), ('partition_hash', 'other')])
def test_each_bound_identity_has_a_contextual_integrity_error(field, value):
    p = _prov()
    with pytest.raises(ModelIntegrityError) as caught:
        verify_provenance({**p.to_dict(), field: value}, p)
    assert caught.value.context['fields'] == [field]
    assert isinstance(caught.value, ValueError) and isinstance(caught.value, ArtifactIntegrityError)


@pytest.mark.parametrize('saved', [None, [], {}, {'feature_order': 'features'},
    {'feature_order': ['repeat', 'repeat']}, {'feature_order': [['nested']]}, {'bank_checksum': ''}])
def test_missing_or_malformed_provenance_is_not_comparable(saved):
    p = _prov()
    value = {**p.to_dict(), **saved} if isinstance(saved, dict) and saved else saved
    with pytest.raises(ModelIntegrityError): verify_provenance(value, p)


@pytest.mark.parametrize('field,value', [('bank_checksum', ''), ('feature_order', ()), ('class_map_hash', None)])
def test_malformed_current_identity_is_a_configuration_error(field, value):
    from dataclasses import replace
    p = _prov()
    with pytest.raises(ConfigurationError): verify_provenance(p.to_dict(), replace(p, **{field: value}))


def test_audit_fields_remain_recorded_without_becoming_runtime_version_pins():
    p = _prov()
    value = {**p.to_dict(), 'seed': 999, 'software_versions': {'torch': 'historical'},
             'training_manifest_hash': 'other', 'loss_mask_config': {'historical': True}}
    assert verify_provenance(value, p) is True


def test_provenance_records_absent_optional_numbering_software(monkeypatch):
    from alignair.applicability import checkpoint
    version = checkpoint.version
    def installed_version(name):
        if name == 'anarcii': raise checkpoint.PackageNotFoundError(name)
        return version(name)
    monkeypatch.setattr(checkpoint, 'version', installed_version)
    assert _prov().software_versions['anarcii'] == 'n/a'


def test_provenance_discovery_does_not_import_optional_hmm_or_numbering_libraries(monkeypatch):
    import builtins
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in ('pyhmmer', 'anarcii'): pytest.fail('version discovery must not import optional engines')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    assert set(_prov().software_versions) == {'torch', 'pyhmmer', 'anarcii'}


def test_loading_checks_both_training_contract_and_runtime_identity(tmp_path):
    import torch
    p = _prov(); path = tmp_path/'state.pt'
    save_checkpoint(path, {'training_contract': 'run', 'provenance': p.to_dict(), 'model': {'w': torch.tensor([1.])}})
    loaded = load_checkpoint(path, training_contract='run', provenance=p)
    assert loaded['model']['w'].item() == 1
    with pytest.raises(ArtifactIntegrityError, match='does not match'):
        load_checkpoint(path, training_contract='other', provenance=p)
    with pytest.raises(ConfigurationError): load_checkpoint(path, training_contract='', provenance=p)
    save_checkpoint(path, {'training_contract': 'run'})
    with pytest.raises(ModelIntegrityError): load_checkpoint(path, training_contract='run', provenance=p)


@pytest.mark.parametrize('payload', [None, [], 1])
def test_a_checkpoint_must_be_a_mapping(tmp_path, payload):
    import torch
    path = tmp_path/'bad.pt'; torch.save(payload, path)
    with pytest.raises(ModelFormatError): load_checkpoint(path, training_contract='run', provenance=_prov())


def test_missing_checkpoint_io_is_chained_with_path(tmp_path):
    path = tmp_path/'missing.pt'
    with pytest.raises(OperationIOError) as caught: load_checkpoint(path, training_contract='run', provenance=_prov())
    assert caught.value.context['path'] == str(path)
    assert isinstance(caught.value.__cause__, FileNotFoundError)


def test_invalid_expected_identity_stops_before_deserialization(tmp_path, monkeypatch):
    import torch
    def forbidden(*args, **kwargs): pytest.fail('invalid expected identity must fail before unpickling')
    monkeypatch.setattr(torch, 'load', forbidden)
    with pytest.raises(ConfigurationError):
        load_checkpoint(tmp_path/'source.pt', training_contract='run', provenance=None)


def test_device_load_failure_retains_device_and_original_cause(tmp_path, monkeypatch):
    import torch
    from alignair.errors import ModelError
    error = RuntimeError('device unavailable during loading')
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(torch, 'load', fail)
    with pytest.raises(ModelError) as caught:
        load_checkpoint(tmp_path/'source.pt', training_contract='run', provenance=_prov(), device='cuda:1')
    assert caught.value.__cause__ is error and caught.value.context['device'] == 'cuda:1'
    assert not isinstance(caught.value, ModelFormatError)


def test_load_preserves_cuda_out_of_memory_for_operational_handling(tmp_path, monkeypatch):
    import torch
    error = torch.cuda.OutOfMemoryError('injected allocation failure')
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(torch, 'load', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught:
        load_checkpoint(tmp_path/'source.pt', training_contract='run', provenance=_prov())
    assert caught.value is error


def test_save_preserves_cuda_out_of_memory_and_the_previous_file(tmp_path, monkeypatch):
    import torch
    path = tmp_path/'state.pt'; path.write_bytes(b'previous')
    error = torch.cuda.OutOfMemoryError('injected allocation failure')
    def fail(*args, **kwargs): raise error
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(torch.cuda.OutOfMemoryError) as caught: save_checkpoint(path, {})
    assert caught.value is error and path.read_bytes() == b'previous' and list(tmp_path.iterdir()) == [path]


def test_sampling_plan_is_checked_independently_of_the_supplied_run_hash(tmp_path):
    path = tmp_path/'checkpoint.pt'; p = _prov(); plan = {'policy': 'v2', 'batch_size': 4}
    save_checkpoint(path, {'training_contract': 'run', 'provenance': p.to_dict(), 'sampling_plan': plan})
    assert load_checkpoint(path, training_contract='run', provenance=p, sampling_plan=plan)['sampling_plan'] == plan
    with pytest.raises(ArtifactIntegrityError):
        load_checkpoint(path, training_contract='run', provenance=p, sampling_plan={**plan, 'batch_size': 3})
    with pytest.raises(ConfigurationError):
        load_checkpoint(path, training_contract='run', provenance=p, sampling_plan={'unsupported': object()})
    save_checkpoint(path, {'training_contract': 'run', 'provenance': p.to_dict(), 'sampling_plan': object()})
    with pytest.raises(ArtifactIntegrityError):
        load_checkpoint(path, training_contract='run', provenance=p, sampling_plan=plan)


@pytest.mark.parametrize('data', [b'', b'not a torch checkpoint'])
def test_corruption_is_a_chained_format_error(tmp_path, data):
    path = tmp_path/'broken.pt'; path.write_bytes(data)
    with pytest.raises(ModelFormatError) as caught: load_checkpoint(path, training_contract='run', provenance=_prov())
    assert caught.value.context['path'] == str(path) and caught.value.__cause__ is not None


@pytest.mark.parametrize('failure', [OSError, RuntimeError])
def test_failed_guard_save_keeps_predecessor_and_cleans_temporary(tmp_path, monkeypatch, failure):
    from pathlib import Path
    import torch
    path = tmp_path/'state.pt'; path.write_bytes(b'previous')
    def fail(state, temporary):
        Path(temporary).write_bytes(b'partial')
        raise failure('interrupted write')
    monkeypatch.setattr(torch, 'save', fail)
    with pytest.raises(OperationIOError) as caught: save_checkpoint(path, {'model': {}})
    assert isinstance(caught.value.__cause__, failure) and caught.value.context['path'] == str(path)
    assert path.read_bytes() == b'previous' and list(tmp_path.iterdir()) == [path]
