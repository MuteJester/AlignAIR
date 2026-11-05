"""The screens' rig (``rig.build``) and the registered baseline screen of its protocol, each made once per session."""
import pytest

from tests.alignair_lab import rig as rigging


@pytest.fixture(scope='session')
def rig(tmp_path_factory):
    return rigging.build(tmp_path_factory.mktemp('rig'))


@pytest.fixture(scope='session')
def baseline(rig):
    """The rig's registered baseline screen, ``base0``: the small lab-null-v2 of init seed 0, cosine schedule."""
    from alignair_lab.cli import main
    if not (rig.screens / 'base0' / 'screen.json').is_file():
        assert main(rigging.baseline_arguments(rig, 'base0')) == 0
    return rig.screens / 'base0'
