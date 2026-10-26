import io
import urllib.error

import pytest

from alignair.aa.guard.data_sources.http import HttpClient
from alignair.errors import ArtifactIntegrityError, DataContractError, OperationIOError

pytestmark = pytest.mark.component


def test_prefix_is_bounded_and_cache_is_bound_to_url_and_limit(tmp_path):
    calls, responses = [], []

    def opener(request, **kwargs):
        calls.append(request.full_url)
        stream = io.BytesIO(b"abcdefghijk")
        responses.append(stream)
        return stream

    client = HttpClient(opener=opener, sleep=lambda _: None)
    options = dict(byte_limit=5, cache_dir=tmp_path)
    assert client.prefix("https://source.test/a", **options) == b"abcde"
    assert client.prefix("https://source.test/a", **options) == b"abcde"
    assert len(calls) == 1 and responses[0].closed
    assert client.prefix("https://source.test/b", **options) == b"abcde"
    assert client.prefix("https://source.test/a", byte_limit=3, cache_dir=tmp_path) == b"abc"
    assert len(calls) == 3


def test_changed_cached_bytes_are_refused_without_refetch(tmp_path):
    calls = []
    client = HttpClient(opener=lambda *a, **kw: calls.append(1) or io.BytesIO(b"data"))
    client.prefix("https://source.test/a", byte_limit=5, cache_dir=tmp_path)
    path = next(tmp_path.glob("*.part"))
    path.write_bytes(b"changed")
    with pytest.raises(ArtifactIntegrityError):
        client.prefix("https://source.test/a", byte_limit=5, cache_dir=tmp_path)
    assert calls == [1]
    assert path.read_bytes() == b"changed"


def test_incomplete_cache_write_is_downloaded_again(tmp_path):
    calls = []
    client = HttpClient(opener=lambda *a, **kw: calls.append(1) or io.BytesIO(b"data"))
    client.prefix("https://source.test/a", byte_limit=5, cache_dir=tmp_path)
    next(tmp_path.glob("*.part.json")).unlink()
    assert client.prefix("https://source.test/a", byte_limit=5, cache_dir=tmp_path) == b"data"
    assert len(calls) == 2


@pytest.mark.parametrize("status, attempts, retryable", [(503, 3, True), (404, 1, False)])
def test_only_transient_http_failures_are_retried(status, attempts, retryable):
    calls, sleeps = [], []

    def opener(request, **kwargs):
        calls.append(1)
        raise urllib.error.HTTPError(request.full_url, status, "unavailable", {}, None)

    client = HttpClient(opener=opener, sleep=sleeps.append, retries=3)
    with pytest.raises(OperationIOError) as exc:
        client.get("https://source.test/a")
    assert len(calls) == attempts
    assert len(sleeps) == attempts - 1
    assert exc.value.retryable is retryable
    assert exc.value.__cause__ is not None


def test_listing_size_limit_is_enforced():
    client = HttpClient(opener=lambda *a, **kw: io.BytesIO(b"a" * 100), max_listing_bytes=10)
    with pytest.raises(DataContractError, match="byte limit"):
        client.get("https://source.test/a")


def test_custom_range_headers_are_sent_and_bound_to_the_cache(tmp_path):
    calls = []

    def opener(request, **kwargs):
        requested = request.get_header("Range")
        calls.append((requested, request.get_header("Referer")))
        return io.BytesIO(b"abcdef" if requested == "bytes=0-4" else b"bcdefg")

    first = HttpClient(opener=opener, headers={"Range": "bytes=0-4", "Referer": "https://source.test/"})
    second = HttpClient(opener=opener, headers={"Range": "bytes=1-5", "Referer": "https://source.test/"})
    options = dict(byte_limit=5, cache_dir=tmp_path)
    assert first.prefix("https://source.test/a", **options) == b"abcde"
    assert second.prefix("https://source.test/a", **options) == b"bcdef"
    assert first.prefix("https://source.test/a", **options) == b"abcde"
    assert calls == [("bytes=0-4", "https://source.test/"), ("bytes=1-5", "https://source.test/")]
