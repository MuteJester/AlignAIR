"""Fresh workspace ownership and strict artifact publication, independent of training."""
from pathlib import Path

import pytest

from alignair.aa.convergence import artifacts
from alignair.errors import ConfigurationError, InputError, OperationIOError

pytestmark = pytest.mark.component


@pytest.mark.parametrize('field', ['art_dir', 'ckpt_dir'])
def test_existing_outputs_are_refused_and_preserved(tmp_path, field):
    paths = {'art_dir': tmp_path/'art', 'ckpt_dir': tmp_path/'ckpt'}
    paths[field].mkdir()
    sentinel = paths[field]/'completed'
    sentinel.write_text('existing work')
    with pytest.raises(ConfigurationError) as caught:
        artifacts.StudyArtifacts(**paths)
    assert caught.value.context['field'] == field
    assert sentinel.read_text() == 'existing work'
    assert not paths['ckpt_dir' if field == 'art_dir' else 'art_dir'].exists()


@pytest.mark.parametrize('paths', [(None, 'ckpt'), ('', 'ckpt'), ('same', 'same'),
                                  ('outer', 'outer/inner'), ('outer/inner', 'outer')])
def test_invalid_or_overlapping_destinations_are_rejected(tmp_path, paths):
    art, ckpt = [tmp_path/value if value else value for value in paths]
    with pytest.raises(ConfigurationError): artifacts.StudyArtifacts(art, ckpt)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize('operation', ['resolve', 'exists'])
def test_path_inspection_failure_is_contextual(tmp_path, monkeypatch, operation):
    def fail(*args): raise OSError('cannot resolve')
    with monkeypatch.context() as patch:
        patch.setattr(Path, operation, fail)
        with pytest.raises(OperationIOError) as caught:
            artifacts.StudyArtifacts(tmp_path/'art', tmp_path/'ckpt')
    assert caught.value.context['field'] == 'art_dir' and isinstance(caught.value.__cause__, OSError)


def test_a_dangling_output_symlink_is_refused_without_following_or_replacing_it(tmp_path):
    path = tmp_path/'art'
    path.symlink_to(tmp_path/'absent', target_is_directory=True)
    with pytest.raises(ConfigurationError): artifacts.StudyArtifacts(path, tmp_path/'ckpt')
    assert path.is_symlink() and not (tmp_path/'absent').exists() and not (tmp_path/'ckpt').exists()


@pytest.mark.parametrize('contaminate_owned', [False, True])
def test_late_claim_collision_preserves_other_work_and_cleans_only_empty_owned_root(tmp_path, monkeypatch, contaminate_owned):
    out = artifacts.StudyArtifacts(tmp_path/'art', tmp_path/'ckpt')
    original = Path.mkdir
    def claim(path, *args, **kwargs):
        if path == out.ckpt_dir:
            original(path)
            (path/'other-run').write_text('keep')
            if contaminate_owned: (out.art_dir/'external-file').write_text('keep')
            raise FileExistsError('another run claimed output')
        return original(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'mkdir', claim)
    with pytest.raises(OperationIOError): out.create()
    assert (out.ckpt_dir/'other-run').read_text() == 'keep'
    assert out.art_dir.exists() == contaminate_owned
    if contaminate_owned: assert (out.art_dir/'external-file').read_text() == 'keep'


def test_publication_requires_claimed_outputs_and_safe_leaf_names(tmp_path):
    out = artifacts.StudyArtifacts(tmp_path/'art', tmp_path/'ckpt')
    with pytest.raises(ConfigurationError): out.write_text('report', 'text')
    out.create()
    for name in ('', '..', '../outside', str(tmp_path/'absolute'), None):
        with pytest.raises(ConfigurationError): out.write_text(name, 'text')
    out.write_text('predeclaration.md', '§5\n')
    assert (out.art_dir/'predeclaration.md').read_text() == '§5\n'


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), object()])
def test_json_encoding_failure_preserves_a_completed_file(tmp_path, bad):
    out = artifacts.StudyArtifacts(tmp_path/'art', tmp_path/'ckpt'); out.create()
    out.write_json('metrics.json', {'score': .5})
    path = out.art_dir/'metrics.json'; previous = path.read_bytes()
    with pytest.raises(InputError): out.write_json('metrics.json', {'score': bad})
    assert path.read_bytes() == previous and list(out.art_dir.iterdir()) == [path]


def test_os_publication_failure_is_chained_with_destination_context(tmp_path, monkeypatch):
    out = artifacts.StudyArtifacts(tmp_path/'art', tmp_path/'ckpt'); out.create()
    def fail(*args, **kwargs): raise OSError('disk full')
    monkeypatch.setattr(artifacts, 'atomic_write_text', fail)
    with pytest.raises(OperationIOError) as caught: out.write_text('report', 'text')
    assert caught.value.context['path'] == str(out.art_dir/'report')
    assert isinstance(caught.value.__cause__, OSError)
