"""One file digest for the provenance sidecar, the model cache, the bundle manifest and the catalog check."""
import hashlib

import pytest

from alignair.io.files import sha256_file

pytestmark = pytest.mark.component


def test_a_file_digest_is_the_sha256_of_its_bytes_read_in_chunks(tmp_path):
    data = bytes(range(256)) * 5000                      # more than one 1 MiB chunk
    path = tmp_path / "artifact.bin"
    path.write_bytes(data)
    assert sha256_file(path) == sha256_file(str(path)) == hashlib.sha256(data).hexdigest()
    (tmp_path / "empty").write_bytes(b"")
    assert sha256_file(tmp_path / "empty") == hashlib.sha256(b"").hexdigest()
