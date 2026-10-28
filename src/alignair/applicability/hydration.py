"""Reusable per-source hydration infrastructure.

Every fresh-cohort source goes through the same durable, safe scaffolding: bounded retry/backoff on the
network call, an immutable sha256 checksum of the raw response, an atomic normalized cache (resumable —
a completed source is never re-fetched), denylist + Phase-3 filtering, and a counts-only source census.

No network code lives here: the caller injects a `fetch` callable and a `normalize` callable, so the
infrastructure is fully testable offline and identical across sources.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def atomic_write_bytes(path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)   # atomic rename on the same filesystem


def atomic_write_text(path, text: str) -> None:
    atomic_write_bytes(path, text.encode())


def fetch_with_retry(fetch, *, retries: int = 4, backoff: float = 1.5, sleep=time.sleep):
    """Call `fetch()` (no args) with bounded exponential backoff; re-raise the last error after `retries`.

    The Gate-3 hydration agent crashed on transient API 500s — this is the guard against that.
    """
    last: Exception | None = None
    for attempt in range(retries):
        try:
            return fetch()
        except Exception as exc:   # transient network/API errors
            last = exc
            if attempt < retries - 1:
                sleep(backoff ** attempt)
    raise last


def filter_disjoint(records, *, denylist, phase3):
    """Drop records overlapping the verified Gate-3 denylist or the Phase-3 therapeutics. Returns
    (kept, dropped_counts)."""
    p3_names, p3_vh = phase3
    p3_names = p3_names or {}
    p3_vh = p3_vh or {}
    seq = denylist.get("sequence_hashes", set())
    lin = denylist.get("lineages", set())
    don = denylist.get("donors", set())
    std = denylist.get("studies", set())

    kept: list = []
    dropped = {"seqhash": 0, "lineage": 0, "phase3": 0, "donor": 0, "study": 0}
    for r in records:
        h = sha256_text(r["aa"])
        nm = (r.get("lineage_key") or "").strip().lower()
        if h in seq:
            dropped["seqhash"] += 1
        elif r.get("lineage_key") in lin:
            dropped["lineage"] += 1
        elif nm in p3_names or h in p3_vh:
            dropped["phase3"] += 1
        elif r.get("donor") and r["donor"] in don:
            dropped["donor"] += 1
        elif r.get("study") and r["study"] in std:
            dropped["study"] += 1
        else:
            kept.append(r)
    return kept, dropped


@dataclass
class SourceResult:
    source: str
    kept: list
    census: dict


def hydrate_source(*, source, store, fetch, normalize, denylist, phase3, provenance,
                   retries: int = 4, backoff: float = 1.5, sleep=time.sleep) -> SourceResult:
    """Hydrate one source end-to-end with checksums / atomic caches / retry / resumability / census.

    `fetch()` -> raw bytes (or JSON-able); `normalize(raw)` -> list of records each with at least
    id/aa/group/category/lineage_key. Writes under `store`:
      - `<source>.raw` + `<source>.raw.sha256` (immutable raw response + checksum)
      - `<source>.normalized.json` (atomic; the resumable cache — kept records only)
      - `<source>.census.json` (counts-only provenance)
    """
    store = Path(store)
    store.mkdir(parents=True, exist_ok=True)
    norm_path = store / f"{source}.normalized.json"
    census_path = store / f"{source}.census.json"

    # Resumability: a completed source (normalized cache + census present) is reused, never re-fetched.
    if norm_path.exists() and census_path.exists():
        kept = json.loads(norm_path.read_text())
        census = json.loads(census_path.read_text())
        return SourceResult(source=source, kept=kept, census=census)

    raw = fetch_with_retry(fetch, retries=retries, backoff=backoff, sleep=sleep)
    raw_bytes = raw if isinstance(raw, (bytes, bytearray)) else json.dumps(raw, sort_keys=True).encode()
    raw_sha = sha256_bytes(raw_bytes)
    atomic_write_bytes(store / f"{source}.raw", raw_bytes)
    atomic_write_text(store / f"{source}.raw.sha256", raw_sha)

    records = normalize(raw)
    kept, dropped = filter_disjoint(records, denylist=denylist, phase3=phase3)

    by_category: dict = {}
    for r in kept:
        by_category[r["category"]] = by_category.get(r["category"], 0) + 1
    census = {
        "source": source,
        "provenance": provenance,
        "raw_sha256": raw_sha,
        "fetched": len(records),
        "kept_after_denylist": len(kept),
        "kept_lineages": len({r["lineage_key"] for r in kept}),
        "dropped": dropped,
        "by_category": by_category,
    }

    atomic_write_text(norm_path, json.dumps(kept))
    atomic_write_text(census_path, json.dumps(census, indent=2, sort_keys=True))
    return SourceResult(source=source, kept=kept, census=census)
