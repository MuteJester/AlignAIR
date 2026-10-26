"""Bounded HTTP reads and verified prefix caches for research sources."""
from __future__ import annotations

import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from alignair.aa.guard.hydration import atomic_write_bytes, atomic_write_text, sha256_bytes, sha256_text
from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError, OperationIOError


def is_transient(error):
    if isinstance(error, urllib.error.HTTPError):
        return error.code in (408, 425, 429, 500, 502, 503, 504)
    return isinstance(error, (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError))


@dataclass
class HttpClient:
    """Inject ``opener`` and ``sleep`` to run adapters without network access."""

    opener: object = None
    sleep: object = time.sleep
    retries: int = 4
    timeout: float = 90
    max_listing_bytes: int = 8 << 20
    headers: dict | None = None

    def _read(self, url, limit):
        if self.retries < 1 or limit < 1:
            raise ConfigurationError("HTTP retries and byte limits must be positive.")
        request = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (AlignAIR fresh-cohort recon)", **(self.headers or {}),
        })
        for attempt in range(self.retries):
            try:
                opener = self.opener or urllib.request.urlopen
                with opener(request, timeout=self.timeout) as response:
                    chunks, remaining = [], limit
                    while remaining:
                        chunk = response.read(remaining)
                        if not chunk:
                            break
                        chunks.append(chunk)
                        remaining -= len(chunk)
                    return b"".join(chunks)
            except (urllib.error.URLError, OSError) as exc:
                retryable = is_transient(exc)
                if isinstance(exc, urllib.error.HTTPError):
                    exc.close()
                if retryable and attempt + 1 < self.retries:
                    self.sleep(1.5 ** attempt)
                    continue
                raise OperationIOError(
                    "Cannot read research source.",
                    context={"url": url, "http_status": getattr(exc, "code", None)},
                    retryable=retryable,
                ) from exc

    def get(self, url):
        raw = self._read(url, self.max_listing_bytes + 1)
        if len(raw) > self.max_listing_bytes:
            raise DataContractError("Research source listing exceeds its byte limit.", context={"url": url})
        return raw

    def prefix(self, url, *, byte_limit, cache_dir=None, rate_s=0):
        """Read at most ``byte_limit`` compressed bytes, validating cached bytes.

        Cache keys include the full URL and byte limit. Incomplete cache writes
        are fetched again; bytes with an invalid checksum are refused.
        """
        cache = metadata = None
        identity = {"url": url, "byte_limit": byte_limit}
        request_key = sha256_text(url)
        if self.headers:
            identity["headers_sha256"] = sha256_text(json.dumps(self.headers, sort_keys=True))
            request_key = sha256_text(url + identity["headers_sha256"])
        if cache_dir is not None:
            cache = Path(cache_dir) / f"{request_key}.{byte_limit}.part"
            metadata = cache.with_suffix(cache.suffix + ".json")
            if cache.exists() and metadata.exists():
                raw = cache.read_bytes()
                try:
                    recorded = json.loads(metadata.read_text())
                except json.JSONDecodeError as exc:
                    raise ArtifactIntegrityError("Invalid download cache metadata.", context={"path": metadata}) from exc
                if (not isinstance(recorded, dict) or any(recorded.get(k) != v for k, v in identity.items())
                        or recorded.get("sha256") != sha256_bytes(raw) or len(raw) > byte_limit):
                    raise ArtifactIntegrityError("Research download cache identity or checksum changed.", context={"path": cache})
                return raw
        raw = self._read(url, byte_limit)
        if cache is not None:
            atomic_write_bytes(cache, raw)
            atomic_write_text(metadata, json.dumps({**identity, "sha256": sha256_bytes(raw)}, sort_keys=True))
        if rate_s:
            self.sleep(rate_s)
        return raw
