"""Verified real-data downloads (``real.fetch``): reads at full quality where AWS ODP has them, genotype files.

The offline tests use synthetic gzipped FASTQs and files, monkeypatched pins, a monkeypatched downloader and
container step, and a temporary data root; nothing is downloaded and no container runs. The gated tests:

- ``ALIGNAIR_ASSAY_DATA`` (the PRJNA555323 data root): the pins against the root's metadata and its files, verified
  in place through a temporary root of links, so nothing is written to the real root;
- with ``ALIGNAIR_ASSAY_NETWORK=1`` too: the IGenotyper ``gene_coords.bed`` at its pinned commit and the manifest;
- with ``ALIGNAIR_ASSAY_DOCKER=1`` as well: W-97's ODP object (34 MB) downloaded, validated and converted in the
  pinned sra-tools container, and its quality alphabet checked.
"""
from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
import urllib.error
import urllib.parse
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from alignair_assay.core import container
from alignair_assay.core.container import Mount, hardened_run
from alignair_assay.real import fetch
from alignair_assay.real.fetch import (
    PILOT_DONORS,
    PILOT_SPLIT,
    SRA_TOOLS_IMAGE,
    VDJBASE_SAMPLE,
    FetchError,
    FetchRecord,
    FileSpec,
    NotFound,
    Refused,
    download,
    ena_fastq_url,
    fetch_file,
    fetch_genotype_files,
    fetch_manifest,
    fetch_reads,
    genotype_specs,
    md5_file,
    quality_kind,
    s3_etag,
    sra_commands,
)

pytestmark = pytest.mark.component

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
NETWORK = os.environ.get("ALIGNAIR_ASSAY_NETWORK") == "1"
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
needs_data = pytest.mark.skipif(not DATA, reason="set ALIGNAIR_ASSAY_DATA to the PRJNA555323 data root")
needs_network = pytest.mark.skipif(not NETWORK, reason="set ALIGNAIR_ASSAY_NETWORK=1 to download")
needs_docker = pytest.mark.skipif(not DOCKER, reason="set ALIGNAIR_ASSAY_DOCKER=1 to run the sra-tools container")

FULL = "#-./0123456789:;<=>?@ABCDEFGHI"
RUN, DONOR = "SRR23654538", "W-97"


def _fastq(n: int, quality, *, mate: int = 1, length: int = 30, run: str = RUN) -> bytes:
    """``n`` records named as ENA names them, each with ``quality(index, length)`` as its quality string."""
    bases = "ACGT"
    lines = []
    for index in range(1, n + 1):
        sequence = "".join(bases[(index * 7 + j * (mate + 2)) % 4] for j in range(length))
        lines += [f"@{run}.{index} {index}/{mate}", sequence, "+", quality(index, length)]
    return ("\n".join(lines) + "\n").encode()


def _lite(index: int, length: int) -> str:
    return "".join("$" if (index + j) % 11 == 0 else "?" for j in range(length))


def _full(index: int, length: int) -> str:
    return "".join(FULL[(index * 3 + j) % len(FULL)] for j in range(length))


def _gz(path: Path, data: bytes) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.GzipFile(path, "wb", mtime=0) as handle:
        handle.write(data)
    return path


def _md5(data: bytes) -> str:
    return hashlib.md5(data).hexdigest()


# --- the quality alphabet ----------------------------------------------------------------------------------------

def test_quality_kind_lite(tmp_path):
    constant = _gz(tmp_path / "constant.fastq.gz", _fastq(50, lambda i, n: "?" * n))
    assert quality_kind(constant) == "lite"
    # SRA-Lite writes Q30 ("?") for a passing base and Q3 ("$") for a rejected one.
    binary = _gz(tmp_path / "binary.fastq.gz", _fastq(50, _lite))
    assert quality_kind(binary) == "lite"


def test_quality_kind_full(tmp_path):
    assert quality_kind(_gz(tmp_path / "full.fastq.gz", _fastq(50, _full))) == "full"
    # One varied record among Lite ones is a full-quality file.
    mixed = _gz(tmp_path / "mixed.fastq.gz", _fastq(50, lambda i, n: _full(i, n) if i == 40 else "?" * n))
    assert quality_kind(mixed) == "full"


def test_quality_kind_reads_only_the_first_n_records(tmp_path):
    late = _gz(tmp_path / "late.fastq.gz", _fastq(30, lambda i, n: _full(i, n) if i > 20 else "?" * n))
    assert quality_kind(late, n=20) == "lite"
    assert quality_kind(late, n=21) == "full"


def test_quality_kind_refuses_what_is_not_a_fastq(tmp_path):
    with pytest.raises(ValueError, match="no record"):
        quality_kind(_gz(tmp_path / "empty.fastq.gz", b""))
    with pytest.raises(ValueError, match="not a FASTQ record"):
        quality_kind(_gz(tmp_path / "bad.fastq.gz", b"@r1\nACGT\n-\nIIII\n"))
    with pytest.raises(ValueError, match="length"):
        quality_kind(_gz(tmp_path / "short.fastq.gz", b"@r1\nACGT\n+\nIII\n"))
    truncated = tmp_path / "truncated.fastq.gz"
    whole = _gz(tmp_path / "whole.fastq.gz", _fastq(2000, _full)).read_bytes()
    truncated.write_bytes(whole[: len(whole) // 2])
    with pytest.raises(ValueError, match="truncated|gzip"):
        quality_kind(truncated, n=2000)
    plain = tmp_path / "plain.fastq.gz"
    plain.write_bytes(_fastq(3, _full))
    with pytest.raises(ValueError, match="gzip"):
        quality_kind(plain)


# --- the pilot ---------------------------------------------------------------------------------------------------

def test_pilot_split_covers_all_pilot_donors_once():
    donors = [donor for split in ("dev", "test") for donor in PILOT_SPLIT[split]]
    assert sorted(donors) == sorted(PILOT_DONORS) and len(donors) == len(set(donors)) == 8
    assert set(PILOT_SPLIT) == {"dev", "test"}
    assert PILOT_SPLIT["dev"] == ("SC-23", "SC-11", "W-97", "W-133")
    assert set(VDJBASE_SAMPLE) == set(PILOT_DONORS)
    assert len(set(PILOT_DONORS.values())) == 8 and len(set(VDJBASE_SAMPLE.values())) == 8
    assert PILOT_DONORS["SC-23"] == "SRR23446586" and PILOT_DONORS["W-97"] == "SRR23654538"
    assert VDJBASE_SAMPLE["SC-23"] == "P25_I58_S1" and VDJBASE_SAMPLE["W-97"] == "P25_I151_S1"
    for run in PILOT_DONORS.values():
        assert run in fetch.READ_PAIRS and run in fetch.ODP_OBJECT and run in fetch.ENA_FASTQ


def test_the_sra_tools_image_is_pinned_by_digest():
    name, _, digest = SRA_TOOLS_IMAGE.partition("@")
    assert name.startswith("ncbi/sra-tools:") and name.split(":")[1][0].isdigit()
    assert digest.startswith("sha256:") and len(digest) == 71 and all(c in "0123456789abcdef" for c in digest[7:])


def test_ena_fastq_url_follows_enas_directory_rule():
    assert ena_fastq_url("SRR23446586", "SRR23446586_1.fastq.gz") == (
        "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR234/086/SRR23446586/SRR23446586_1.fastq.gz")
    assert ena_fastq_url("SRR1234567", "x") == "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/007/SRR1234567/x"
    assert ena_fastq_url("SRR123456", "x") == "https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR123/SRR123456/x"


def test_unknown_runs_and_donors_are_refused(tmp_path):
    with pytest.raises(ValueError, match="not a pilot run"):
        fetch_reads("SRR00000001", tmp_path)
    with pytest.raises(ValueError, match="not a pilot donor"):
        fetch_genotype_files("W-1", tmp_path)


# --- verified files ----------------------------------------------------------------------------------------------

def _spec(data: bytes, path: str = "genotypes/source_a/file.csv", **overrides) -> FileSpec:
    fields = dict(path=path, url="https://example.org/file.csv", md5=_md5(data), bytes=len(data), kind="test",
                  detail={"note": "synthetic"})
    return FileSpec(**{**fields, **overrides})


def _no_download(*args, **kwargs):
    raise AssertionError(f"the downloader was called: {args}")


def _serving(contents: dict[str, bytes], calls: list | None = None):
    """A downloader that writes ``contents[url]`` to the destination and records each call."""
    def fake(url, dest, *, timeout=None, retries=None):
        if calls is not None:
            calls.append(url)
        if url not in contents:
            raise NotFound(f"{url}: HTTP 404")
        Path(dest).write_bytes(contents[url])
        return {"url": url, "bytes": len(contents[url]), "md5": _md5(contents[url]), "headers": {}}
    return fake


def test_verified_record_is_reused(tmp_path, monkeypatch):
    data = b"gene,allele\nIGHV1-2,IGHV1-2*02\n"
    spec = _spec(data)
    target = tmp_path / spec.path
    target.parent.mkdir(parents=True)
    target.write_bytes(data)
    (target.parent / "MD5SUMS.txt").write_text(f"{spec.md5}  file.csv\n", encoding="utf-8")
    monkeypatch.setattr(fetch, "download", _no_download)
    record = fetch_file(spec, tmp_path)
    assert record == FetchRecord(path=target, md5=spec.md5, bytes=len(data), source_url=spec.url, kind="test",
                                 quality=None)
    assert record.detail["reused"] is True and record.detail["note"] == "synthetic"
    assert target.read_bytes() == data


def test_md5_mismatch_refuses(tmp_path, monkeypatch):
    data = b"the pinned bytes\n"
    spec = _spec(data)
    target = tmp_path / spec.path
    # A download that is not the pinned file is refused and leaves nothing behind.
    monkeypatch.setattr(fetch, "download", _serving({spec.url: b"other bytes!!!!!\n"}))
    with pytest.raises(FetchError, match="md5"):
        fetch_file(spec, tmp_path)
    assert not target.exists() and list(target.parent.iterdir()) == []
    # A file already there that is not the pinned one is refused and left as it is.
    target.write_bytes(b"changed in place\n")
    monkeypatch.setattr(fetch, "download", _no_download)
    with pytest.raises(FetchError, match="md5"):
        fetch_file(spec, tmp_path)
    assert target.read_bytes() == b"changed in place\n"
    # The directory's MD5SUMS.txt must agree with the pin.
    target.write_bytes(data)
    (target.parent / "MD5SUMS.txt").write_text(f"{'0' * 32}  file.csv\n", encoding="utf-8")
    with pytest.raises(FetchError, match="MD5SUMS.txt"):
        fetch_file(spec, tmp_path)


def test_a_missing_file_is_downloaded_verified_and_listed(tmp_path, monkeypatch):
    data = b"igh\t2699\t2761\tIGHJ6\n"
    spec = _spec(data, path="genotypes/igenotyper/gene_coords.bed")
    calls = []
    monkeypatch.setattr(fetch, "download", _serving({spec.url: data}, calls))
    record = fetch_file(spec, tmp_path)
    target = tmp_path / spec.path
    assert calls == [spec.url] and target.read_bytes() == data
    assert record.path == target and record.md5 == spec.md5 and record.detail["reused"] is False
    assert (target.parent / "MD5SUMS.txt").read_text(encoding="utf-8") == f"{spec.md5}  gene_coords.bed\n"
    assert sorted(p.name for p in target.parent.iterdir()) == ["MD5SUMS.txt", "gene_coords.bed"]
    # Fetched again, it is reused.
    monkeypatch.setattr(fetch, "download", _no_download)
    assert fetch_file(spec, tmp_path).detail["reused"] is True


def test_a_wrong_size_is_refused_before_the_md5(tmp_path, monkeypatch):
    data = b"twelve bytes"
    spec = _spec(data, bytes=13)
    monkeypatch.setattr(fetch, "download", _serving({spec.url: data}))
    with pytest.raises(FetchError, match="13 bytes"):
        fetch_file(spec, tmp_path)


def test_genotype_files_are_the_pinned_four(tmp_path, monkeypatch):
    specs = genotype_specs("W-97")
    assert set(specs) == {"vdjbase_csv", "vdjbase_bam", "ogrdb", "gene_coords"}
    assert specs["vdjbase_csv"].path == "genotypes/vdjbase_genomic_samples/P25_I151_S1.csv"
    assert specs["vdjbase_csv"].url == ("https://vdjbase.org/static/study_data/Genomic/samples/Human/IGH/P25/"
                                        "P25_I151_S1/P25_I151_S1.csv")
    assert specs["vdjbase_bam"].md5 == "3a11bea90553228d487ced00b9fc2945"
    assert specs["ogrdb"].path == "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json"
    assert specs["ogrdb"].md5 == "296bf2ef309599827c1932537c45c18c"
    coords = specs["gene_coords"]
    assert coords.path == "genotypes/igenotyper/gene_coords.bed"
    assert fetch.IGENOTYPER_COMMIT in coords.url and coords.url.startswith("https://raw.githubusercontent.com/")
    assert coords.detail["commit"] == fetch.IGENOTYPER_COMMIT and len(fetch.IGENOTYPER_COMMIT) == 40
    assert all(spec.url.startswith("https://") for spec in specs.values())

    # Every file fetched and verified (synthetic bytes pinned in their place), then a record per donor.
    contents = {spec.url: f"{key}\n".encode() for key, spec in specs.items()}
    synthetic = {key: replace(spec, md5=_md5(contents[spec.url]), bytes=len(contents[spec.url]))
                 for key, spec in specs.items()}
    monkeypatch.setattr(fetch, "genotype_specs", lambda donor: synthetic)
    monkeypatch.setattr(fetch, "download", _serving(contents))
    records = fetch_genotype_files("W-97", tmp_path)
    assert set(records) == set(specs) and all(r.quality is None for r in records.values())
    assert records["gene_coords"].path == tmp_path / "genotypes/igenotyper/gene_coords.bed"
    assert json.loads((tmp_path / "fetch/genotypes/W-97.json").read_text(encoding="utf-8"))["gene_coords"]["path"] \
        == "genotypes/igenotyper/gene_coords.bed"
    monkeypatch.setattr(fetch, "download", _no_download)
    assert fetch_genotype_files("W-97", tmp_path) == records
    assert fetch_manifest(tmp_path)["genotypes"]["W-97"]["ogrdb"]["md5"] == synthetic["ogrdb"].md5


# --- the downloader ----------------------------------------------------------------------------------------------

class _Response(io.BytesIO):
    def __init__(self, data: bytes, url: str):
        super().__init__(data)
        self.headers = {"Content-Length": str(len(data)), "ETag": '"abc-2"'}
        self._url = url

    def geturl(self):
        return self._url


def test_download_retries_at_most_twice_with_a_timeout(tmp_path, monkeypatch):
    calls = []

    def flaky(request, timeout):
        calls.append(timeout)
        if len(calls) < 3:
            raise urllib.error.URLError("connection reset")
        return _Response(b"payload", request.full_url)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(fetch, "RETRY_WAIT", 0)
    found = download("https://vdjbase.org/x", tmp_path / "x", timeout=7)
    assert calls == [7, 7, 7] and (tmp_path / "x").read_bytes() == b"payload"
    assert found["md5"] == _md5(b"payload") and found["bytes"] == 7 and found["headers"]["ETag"] == '"abc-2"'

    calls.clear()

    def timing_out(request, timeout):
        calls.append(timeout)
        raise TimeoutError("timed out")

    monkeypatch.setattr(fetch.urllib.request, "urlopen", timing_out)
    with pytest.raises(FetchError, match="3 attempts"):
        download("https://vdjbase.org/y", tmp_path / "y", timeout=7)
    assert len(calls) == 3


def test_download_does_not_retry_a_missing_object(tmp_path, monkeypatch):
    calls = []

    def missing(request, timeout):
        calls.append(request.full_url)
        raise urllib.error.HTTPError(request.full_url, 403, "Forbidden", {}, None)

    monkeypatch.setattr(fetch.urllib.request, "urlopen", missing)
    with pytest.raises(NotFound, match="403"):
        download("https://sra-pub-run-odp.s3.amazonaws.com/sra/x/x", tmp_path / "m")
    assert len(calls) == 1


def test_download_refuses_a_short_body(tmp_path, monkeypatch):
    def short(request, timeout):
        response = _Response(b"abc", request.full_url)
        response.headers["Content-Length"] = "10"
        return response

    monkeypatch.setattr(fetch.urllib.request, "urlopen", short)
    monkeypatch.setattr(fetch, "RETRY_WAIT", 0)
    with pytest.raises(FetchError, match="3 of 10 bytes"):
        download("https://vdjbase.org/short", tmp_path / "s")


def test_download_refuses_plain_http_and_other_hosts_without_retrying(tmp_path, monkeypatch):
    calls = []

    def redirected(to):
        def answer(request, timeout):
            calls.append(request.full_url)
            return _Response(b"payload", to)
        return answer

    monkeypatch.setattr(fetch, "RETRY_WAIT", 0)
    monkeypatch.setattr(fetch.urllib.request, "urlopen", redirected("https://vdjbase.org/x"))
    for url in ("http://vdjbase.org/plain", "https://example.org/elsewhere"):
        with pytest.raises(Refused, match="https|host"):
            download(url, tmp_path / "p")
    assert calls == []  # refused before any request
    # A redirect off https, or to a host that is not a pinned source, is refused at once, not retried.
    for to in ("http://vdjbase.org/x", "https://example.org/x"):
        calls.clear()
        monkeypatch.setattr(fetch.urllib.request, "urlopen", redirected(to))
        with pytest.raises(Refused, match="redirected"):
            download("https://vdjbase.org/x", tmp_path / "r")
        assert len(calls) == 1 and not (tmp_path / "r").exists()
    assert {spec.url.split("/")[2] for spec in genotype_specs("W-97").values()} <= fetch.ALLOWED_HOSTS
    assert {fetch.ODP_URL.split("/")[2], ena_fastq_url(RUN, "x").split("/")[2]} <= fetch.ALLOWED_HOSTS


def test_s3_etag_of_single_and_multipart_objects(tmp_path):
    data = bytes(range(256)) * 41  # 10,496 bytes
    path = tmp_path / "object"
    path.write_bytes(data)
    assert s3_etag(path, part_size=4096) == hashlib.md5(b"".join(
        hashlib.md5(data[i:i + 4096]).digest() for i in range(0, len(data), 4096))).hexdigest() + "-3"
    assert s3_etag(path, part_size=1 << 20, single=True) == _md5(data)


# --- reads: ODP at full quality, else the ENA FASTQs --------------------------------------------------------------

def _pin_run(monkeypatch, root: Path, *, pairs: int, sra: bytes, ena_quality=_lite, ena: bool = True,
             converted=_full, length: int = 30):
    """Pins for ``RUN`` on a synthetic object, its conversion (``converted`` qualities) and synthetic ENA FASTQs
    under ``root``; the pinned image is what ``check_image`` finds."""
    monkeypatch.setitem(fetch.READ_PAIRS, RUN, pairs)
    monkeypatch.setitem(fetch.ODP_OBJECT, RUN, (len(sra), s3_etag_bytes(sra, 16)))
    monkeypatch.setattr(fetch, "S3_PART_SIZE", 16)
    monkeypatch.setitem(fetch.FASTQ_MD5, RUN, tuple(_md5(_fastq(pairs, converted, mate=m, length=length))
                                                    for m in (1, 2)))
    names, pins = (f"{RUN}_1.fastq.gz", f"{RUN}_2.fastq.gz"), []
    for mate, name in enumerate(names, start=1):
        path = root / "airrseq" / DONOR / RUN / name
        data = _gz(path, _fastq(pairs, ena_quality, mate=mate)).read_bytes()
        if not ena:
            path.unlink()
        pins.append((name, len(data), _md5(data)))
    monkeypatch.setitem(fetch.ENA_FASTQ, RUN, tuple(pins))
    monkeypatch.setattr(fetch, "check_image", _found_image)


def _found_image(image=None):
    """``check_image`` as Docker answers it for the pinned image: its RepoDigests hold the pinned digest."""
    reference = image or fetch.SRA_TOOLS_IMAGE
    name, _, digest = reference.partition("@")
    return {"image": reference, "id": "sha256:" + "1" * 64, "repo_digests": [f"{name.rsplit(':', 1)[0]}@{digest}"]}


def s3_etag_bytes(data: bytes, part_size: int) -> str:
    parts = [hashlib.md5(data[i:i + part_size]).digest() for i in range(0, len(data), part_size)]
    return f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"


def _converter(pairs: int, quality, calls: list | None = None, *, mates=(1, 2), length: int = 30, edit=None):
    """A stand-in for the container step: ``fasterq-dump --split-files`` FASTQs in ``out_dir`` (``edit(mate,
    data)`` changes a mate's bytes)."""
    def fake(sra_path, out_dir, *, threads=None):
        if calls is not None:
            calls.append(Path(sra_path))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        for mate in mates:
            data = _fastq(pairs, quality, mate=mate, length=length)
            (Path(out_dir) / f"{RUN}_{mate}.fastq").write_bytes(edit(mate, data) if edit else data)
        return {"commands": [["docker", "run", "fake"]], "vdb_validate": "Database 'x' is consistent"}
    return fake


SRA = b"NCBI.sra synthetic object bytes, more than one part long"


def test_odp_at_full_quality_is_used(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=40, sra=SRA)
    calls, converted = [], []
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}, calls))
    monkeypatch.setattr(fetch, "convert_sra", _converter(40, _full, converted))
    r1, r2, record = fetch_reads(RUN, tmp_path)
    run_dir = tmp_path / "airrseq_odp" / DONOR / RUN
    assert (r1, r2) == (run_dir / f"{RUN}_1.fastq.gz", run_dir / f"{RUN}_2.fastq.gz")
    assert record.quality == "full" and record.kind == "reads" and record.path == run_dir
    assert record.source_url == f"https://sra-pub-run-odp.s3.amazonaws.com/sra/{RUN}/{RUN}"
    assert calls == [record.source_url] and converted == [run_dir / RUN]
    assert quality_kind(r1) == quality_kind(r2) == "full"
    detail = record.detail
    assert detail["source"] == "odp" and detail["read_pairs"] == 40
    assert detail["files"][r1.name]["records"] == 40 and detail["files"][r1.name]["md5"] == md5_file(r1)
    assert [detail["files"][p.name]["fastq_md5"] for p in (r1, r2)] == list(fetch.FASTQ_MD5[RUN])
    assert detail["recipe_sha256"] == fetch.recipe_digest(RUN)
    assert detail["odp"]["etag"] == fetch.ODP_OBJECT[RUN][1] and detail["odp"]["md5"] == _md5(SRA)
    assert detail["ena_agreement"] == {"records": 40, "same": True}
    assert record.bytes == r1.stat().st_size + r2.stat().st_size
    assert record.md5 == _md5(f"{md5_file(r1)}  {r1.name}\n{md5_file(r2)}  {r2.name}\n".encode())
    # Only the object, the pair and its MD5SUMS.txt are left; the uncompressed FASTQs are gone.
    assert sorted(p.name for p in run_dir.iterdir()) == ["MD5SUMS.txt", RUN, r1.name, r2.name]
    listed = (run_dir / "MD5SUMS.txt").read_text(encoding="utf-8").splitlines()
    assert f"{md5_file(r1)}  {r1.name}" in listed and f"{_md5(SRA)}  {RUN}" in listed
    with gzip.open(r1, "rt") as handle:
        assert handle.readline() == f"@{RUN}.1 1/1\n"


def test_verified_reads_record_is_reused_and_a_changed_file_refused(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=12, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    monkeypatch.setattr(fetch, "convert_sra", _converter(12, _full))
    first = fetch_reads(RUN, tmp_path)
    monkeypatch.setattr(fetch, "download", _no_download)
    monkeypatch.setattr(fetch, "convert_sra", _no_download)
    monkeypatch.setattr(fetch, "check_image", _no_download)
    again = fetch_reads(RUN, tmp_path)
    assert again[:2] == first[:2] and again[2] == first[2]
    _gz(first[0], _fastq(12, _lite))
    with pytest.raises(FetchError, match="changed since"):
        fetch_reads(RUN, tmp_path)


def _fetched_once(tmp_path, monkeypatch, pairs: int = 7):
    """``RUN`` fetched from ODP at full quality, and the downloads and conversions made so far."""
    _pin_run(monkeypatch, tmp_path, pairs=pairs, sra=SRA)
    downloads, conversions = [], []
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}, downloads))
    monkeypatch.setattr(fetch, "convert_sra", _converter(pairs, _full, conversions))
    first = fetch_reads(RUN, tmp_path)
    assert len(downloads) == len(conversions) == 1
    return first, downloads, conversions


def test_recipe_digest_follows_what_makes_the_reads_and_nothing_else(monkeypatch):
    before = fetch.recipe_digest(RUN)
    assert before == fetch.recipe_digest(RUN) and len(before) == 64
    assert fetch.recipe_digest("SRR23446586") != before
    # Not the threads, the user, the container's name or the run directory: the argv's placeholders.
    monkeypatch.setattr(fetch, "THREADS", 16)
    monkeypatch.setattr(container.os, "getpid", lambda: 1)
    assert fetch.recipe_digest(RUN) == before
    for name, value in (("GZIP_LEVEL", 9), ("QUALITY_RECORDS", 10), ("ENA_SEQ_DEFLINE", "@$ac.$si"),
                        ("SRA_TOOLS_IMAGE", "ncbi/sra-tools:3.4.1@sha256:" + "a" * 64)):
        with monkeypatch.context() as patch:
            patch.setattr(fetch, name, value)
            assert fetch.recipe_digest(RUN) != before, name
    for table, value in ((fetch.ODP_OBJECT, (1, "x-1")), (fetch.READ_PAIRS, 1), (fetch.FASTQ_MD5, ("a", "b")),
                         (fetch.ENA_FASTQ, (("a", 1, "0" * 32), ("b", 1, "1" * 32)))):
        with monkeypatch.context() as patch:
            patch.setitem(table, RUN, value)
            assert fetch.recipe_digest(RUN) != before


def test_a_changed_pin_forces_reconversion(tmp_path, monkeypatch):
    first, downloads, conversions = _fetched_once(tmp_path, monkeypatch)
    (name, size, md5), second = fetch.ENA_FASTQ[RUN]
    monkeypatch.setitem(fetch.ENA_FASTQ, RUN, ((name, size, "f" * 32), second))
    again = fetch_reads(RUN, tmp_path)
    # The object is reused after its ETag check; only the conversion is redone.
    assert len(downloads) == 1 and len(conversions) == 2
    assert again[2].detail["recipe_sha256"] == fetch.recipe_digest(RUN) != first[2].detail["recipe_sha256"]
    assert "recipe" in again[2].detail["replaced"]["reason"]
    assert again[2].md5 == first[2].md5  # the same reads, deterministically
    fetch_reads(RUN, tmp_path)
    assert len(conversions) == 2  # current now: reused
    monkeypatch.setattr(fetch, "GZIP_LEVEL", 1)
    fetch_reads(RUN, tmp_path)
    assert len(conversions) == 3


def test_a_changed_image_digest_forces_reconversion(tmp_path, monkeypatch):
    first, downloads, conversions = _fetched_once(tmp_path, monkeypatch)
    monkeypatch.setattr(fetch, "SRA_TOOLS_IMAGE", "ncbi/sra-tools:3.4.2@sha256:" + "a" * 64)
    again = fetch_reads(RUN, tmp_path)
    assert len(downloads) == 1 and len(conversions) == 2
    assert again[2].detail["image"]["repo_digests"] == ["ncbi/sra-tools@sha256:" + "a" * 64]


def test_a_record_without_the_pinned_image_or_a_recipe_is_stale(tmp_path, monkeypatch):
    first, downloads, conversions = _fetched_once(tmp_path, monkeypatch)
    record_path = tmp_path / "fetch" / "reads" / f"{RUN}.json"
    # The record says another image ran, though its recipe is current.
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["detail"]["image"]["repo_digests"] = ["ncbi/sra-tools@sha256:" + "b" * 64]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    again = fetch_reads(RUN, tmp_path)
    assert len(conversions) == 2 and "image" in again[2].detail["replaced"]["reason"]
    # A record written before recipes were recorded (the first pilot fetch) is rebuilt once, then reused.
    record = json.loads(record_path.read_text(encoding="utf-8"))
    del record["detail"]["recipe_sha256"]
    record_path.write_text(json.dumps(record), encoding="utf-8")
    fetch_reads(RUN, tmp_path)
    fetch_reads(RUN, tmp_path)
    assert len(conversions) == 3 and len(downloads) == 1


def test_a_lite_fallback_retries_odp_when_it_is_reused(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=5, sra=SRA)
    downloads, conversions = [], []
    monkeypatch.setattr(fetch, "download", _serving({}, downloads))
    monkeypatch.setattr(fetch, "convert_sra", _converter(5, _full, conversions))
    _, _, lite = fetch_reads(RUN, tmp_path)
    assert lite.quality == "lite" and lite.detail["source"] == "ena" and len(downloads) == 1
    # Reused while ODP still has no object: ODP is tried once more, and the Lite record is kept.
    _, _, kept = fetch_reads(RUN, tmp_path)
    assert len(downloads) == 2 and conversions == [] and kept == lite and kept.detail["reused"] is True
    assert kept.detail["odp_retry"]["status"] == "not found"
    # Reused once ODP has it: the record is replaced by the full-quality reads.
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}, downloads))
    r1, _, full = fetch_reads(RUN, tmp_path)
    assert full.quality == "full" and full.detail["source"] == "odp" and len(conversions) == 1
    assert full.detail["replaced"]["md5"] == lite.md5 and r1.parent == tmp_path / "airrseq_odp" / DONOR / RUN
    fetch_reads(RUN, tmp_path)
    assert len(downloads) == 3 and len(conversions) == 1  # a full record is reused without trying anything


def test_odp_at_lite_quality_falls_back_to_the_ena_fastqs(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=25, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    monkeypatch.setattr(fetch, "convert_sra", _converter(25, _lite))
    r1, r2, record = fetch_reads(RUN, tmp_path)
    ena_dir = tmp_path / "airrseq" / DONOR / RUN
    assert (r1, r2) == (ena_dir / f"{RUN}_1.fastq.gz", ena_dir / f"{RUN}_2.fastq.gz")
    assert record.quality == "lite" and record.detail["source"] == "ena" and record.path == ena_dir
    assert record.source_url == f"https://ftp.sra.ebi.ac.uk/vol1/fastq/SRR236/038/{RUN}/"
    assert record.detail["odp"]["status"] == "lite" and "SRA-Lite" in record.detail["fallback_reason"]
    # The Lite conversion is not kept; the object stays as the evidence.
    assert sorted(p.name for p in (tmp_path / "airrseq_odp" / DONOR / RUN).iterdir()) == ["MD5SUMS.txt", RUN]


def test_a_missing_odp_object_falls_back_and_the_ena_quality_is_measured(tmp_path, monkeypatch):
    # The ENA FASTQs carry their own alphabet: SC-23's are full quality, so "full" is what is recorded.
    _pin_run(monkeypatch, tmp_path, pairs=10, sra=SRA, ena_quality=_full)
    monkeypatch.setattr(fetch, "download", _serving({}))
    monkeypatch.setattr(fetch, "convert_sra", _no_download)
    _, _, record = fetch_reads(RUN, tmp_path)
    assert record.detail["source"] == "ena" and record.quality == "full"
    assert record.detail["odp"]["status"] == "not found" and "404" in record.detail["fallback_reason"]
    assert not (tmp_path / "airrseq_odp" / DONOR / RUN).exists()


def test_missing_ena_fastqs_are_downloaded_and_verified(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=8, sra=SRA, ena=False)
    names = [pin[0] for pin in fetch.ENA_FASTQ[RUN]]
    contents = {ena_fastq_url(RUN, name): _gz(tmp_path / "staging" / name, _fastq(8, _lite, mate=m)).read_bytes()
                for m, name in enumerate(names, start=1)}
    monkeypatch.setattr(fetch, "download", _serving(contents))
    _, _, record = fetch_reads(RUN, tmp_path)
    assert record.detail["source"] == "ena" and record.quality == "lite"
    assert all((tmp_path / "airrseq" / DONOR / RUN / name).is_file() for name in names)


def test_a_conversion_with_the_wrong_read_count_is_refused(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=20, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    monkeypatch.setattr(fetch, "convert_sra", _converter(19, _full))
    with pytest.raises(FetchError, match="19 records.*20"):
        fetch_reads(RUN, tmp_path)
    assert not (tmp_path / "fetch" / "reads" / f"{RUN}.json").exists()
    monkeypatch.setattr(fetch, "convert_sra", _converter(20, _full, mates=(1,)))
    with pytest.raises(FetchError, match="pair"):
        fetch_reads(RUN, tmp_path)


def test_odp_reads_that_are_not_the_ena_reads_are_refused(tmp_path, monkeypatch):
    # The conversion is its pins' (31 bases a read), but the ENA FASTQs there have 30: not the same reads.
    _pin_run(monkeypatch, tmp_path, pairs=6, sra=SRA, length=31)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    monkeypatch.setattr(fetch, "convert_sra", _converter(6, _full, length=31))
    with pytest.raises(FetchError, match="ENA"):
        fetch_reads(RUN, tmp_path)


def test_a_conversion_that_is_not_the_pinned_reads_is_refused(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=6, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    other = _converter(6, lambda i, n: _full(i + 1, n))  # full quality, other symbols than the pinned bytes
    monkeypatch.setattr(fetch, "convert_sra", other)
    with pytest.raises(FetchError, match="pinned"):
        fetch_reads(RUN, tmp_path)
    assert not (tmp_path / "fetch" / "reads" / f"{RUN}.json").exists()


def _swap_records(data: bytes, first: int, second: int) -> bytes:
    lines = data.decode().splitlines(keepends=True)
    records = [lines[i:i + 4] for i in range(0, len(lines), 4)]
    records[first], records[second] = records[second], records[first]
    return "".join(line for record in records for line in record).encode()


def test_mates_that_are_not_paired_record_by_record_are_refused(tmp_path, monkeypatch):
    # Records 5 and 6 of mate 2 swapped, past the first QUALITY_RECORDS (2 here) that the ENA comparison reads: the
    # pairing is checked over the whole file.
    _pin_run(monkeypatch, tmp_path, pairs=8, sra=SRA)
    monkeypatch.setattr(fetch, "QUALITY_RECORDS", 2)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    swapped = _converter(8, _full, edit=lambda mate, data: _swap_records(data, 4, 5) if mate == 2 else data)
    monkeypatch.setattr(fetch, "convert_sra", swapped)
    with pytest.raises(FetchError, match=rf"record 5: mate 1 is .*{RUN}\.5.*mate 2 .*{RUN}\.6.*not paired"):
        fetch_reads(RUN, tmp_path)
    # A mate with one record more than the other is refused too.
    longer = _converter(8, _full, edit=lambda mate, data: _fastq(9, _full, mate=2) if mate == 2 else data)
    monkeypatch.setattr(fetch, "convert_sra", longer)
    with pytest.raises(FetchError, match="not paired"):
        fetch_reads(RUN, tmp_path)


def test_a_malformed_quality_in_the_conversion_is_a_fetch_error(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=4, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    short = _converter(4, lambda i, n: _full(i, n - 1))  # a quality string one symbol short of its bases
    monkeypatch.setattr(fetch, "convert_sra", short)
    with pytest.raises(FetchError, match="length"):
        fetch_reads(RUN, tmp_path)


def test_an_odp_object_that_is_not_the_pinned_one_is_refused(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=5, sra=SRA)
    tampered = SRA[:-1] + b"?"
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): tampered}))
    monkeypatch.setattr(fetch, "convert_sra", _no_download)
    with pytest.raises(FetchError, match="ETag"):
        fetch_reads(RUN, tmp_path)
    assert not (tmp_path / "airrseq_odp" / DONOR / RUN / RUN).exists()


def test_docker_missing_refuses_before_any_download(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=5, sra=SRA)

    def no_docker(image=None):
        raise FetchError("docker is not available: install Docker or start its daemon")

    monkeypatch.setattr(fetch, "check_image", no_docker)
    monkeypatch.setattr(fetch, "download", _no_download)
    with pytest.raises(FetchError, match="docker"):
        fetch_reads(RUN, tmp_path)
    assert not (tmp_path / "airrseq_odp").exists()


def test_too_little_free_space_refuses_before_any_download(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=5, sra=SRA)
    monkeypatch.setattr(fetch.shutil, "disk_usage", lambda path: SimpleNamespace(total=10**9, used=10**9, free=100))
    monkeypatch.setattr(fetch, "download", _no_download)
    with pytest.raises(FetchError, match="free"):
        fetch_reads(RUN, tmp_path)


def test_check_image_refuses_another_digest(monkeypatch):
    # the shared container helper checks the image (core.container.check_image); its refusals are fetch errors
    inspected = json.dumps([{"Id": "sha256:" + "2" * 64, "RepoDigests": ["ncbi/sra-tools@sha256:" + "f" * 64]}])
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed(inspected))
    with pytest.raises(FetchError, match="digest"):
        fetch.check_image()
    # an inspection that is not docker's JSON is refused, not a crash
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed("[]"))
    with pytest.raises(FetchError, match="cannot read docker's inspection"):
        fetch.check_image()

    def missing(argv, **kwargs):
        raise FileNotFoundError("docker")

    monkeypatch.setattr(container.subprocess, "run", missing)
    with pytest.raises(FetchError, match="docker is not available"):
        fetch.check_image()
    pinned = json.dumps([{"Id": "sha256:" + "1" * 64,
                          "RepoDigests": [container.pinned_repo_digest(SRA_TOOLS_IMAGE)]}])
    monkeypatch.setattr(container.subprocess, "run", lambda argv, **kwargs: _Completed(pinned))
    assert fetch.check_image()["repo_digests"] == [container.pinned_repo_digest(SRA_TOOLS_IMAGE)]


def test_sra_commands_run_offline_as_the_user_on_the_run_directory_only(tmp_path):
    commands = sra_commands(tmp_path, RUN, threads=3)
    assert [command[command.index(SRA_TOOLS_IMAGE) + 1] for command in commands] == ["vdb-validate", "fasterq-dump"]
    for command in commands:
        assert command[:3] == ["docker", "run", "--rm"]
        assert command[command.index("--name") + 1] == f"alignair-fetch-{RUN}-{os.getpid()}"
        assert command[command.index("--network") + 1] == "none"
        assert command[command.index("--cap-drop") + 1] == "ALL"
        assert command[command.index("--security-opt") + 1] == "no-new-privileges"
        assert command[command.index("--user") + 1] == f"{os.getuid()}:{os.getgid()}"
        # the shared hardened run (core.container.hardened_run): the run directory alone, by a bind mount that
        # must exist (``-v`` would create a missing one as root), writable for the conversion
        assert "-v" not in command
        mounts = [command[i + 1] for i, part in enumerate(command) if part == "--mount"]
        assert mounts == [f"type=bind,source={tmp_path},target=/data"]
        assert command[command.index("-w") + 1] == "/data" and command[command.index("-e") + 1] == "HOME=/tmp"
        assert command == hardened_run(SRA_TOOLS_IMAGE, command[command.index(SRA_TOOLS_IMAGE) + 1:],
                                       name=f"alignair-fetch-{RUN}-{os.getpid()}", workdir="/data",
                                       env={"HOME": "/tmp"}, mounts=[Mount(tmp_path, "/data", read_only=False)])
    dump = commands[1]
    assert "--split-files" in dump and dump[-1] == f"/data/{RUN}" and dump[dump.index("--threads") + 1] == "3"
    assert dump[dump.index("--seq-defline") + 1] == "@$ac.$si $sn/$ri"


class _Completed:
    def __init__(self, stdout="", returncode=0, stderr=""):
        self.stdout, self.stderr, self.returncode = stdout, stderr, returncode


def test_convert_sra_runs_the_hardened_containers(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        return _Completed(stderr="Database 'x' is consistent" if "vdb-validate" in argv else "spots read : 3")

    monkeypatch.setattr(container.subprocess, "run", run)
    found = fetch.convert_sra(tmp_path / RUN, tmp_path / "convert", threads=2)
    assert calls == found["commands"] == sra_commands(tmp_path, RUN, threads=2) and len(calls) == 2
    for argv in calls:
        assert {"--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--name"} <= set(argv)
    assert found["vdb_validate"] == "Database 'x' is consistent"


def test_a_conversion_timeout_kills_its_container(tmp_path, monkeypatch):
    calls = []

    def run(argv, **kwargs):
        calls.append(argv)
        if argv[:2] == ["docker", "run"]:
            raise container.subprocess.TimeoutExpired(argv, kwargs.get("timeout"))
        return _Completed()

    monkeypatch.setattr(container.subprocess, "run", run)
    with pytest.raises(FetchError, match="did not finish.*killed"):
        fetch.convert_sra(tmp_path / RUN, tmp_path / "convert")
    name = f"alignair-fetch-{RUN}-{os.getpid()}"
    assert calls[0][calls[0].index("--name") + 1] == name and calls[1] == ["docker", "kill", name]
    assert len(calls) == 2  # the second container is not started


# --- the manifest ------------------------------------------------------------------------------------------------

def test_fetch_manifest_lists_what_was_fetched(tmp_path, monkeypatch):
    _pin_run(monkeypatch, tmp_path, pairs=9, sra=SRA)
    monkeypatch.setattr(fetch, "download", _serving({fetch.ODP_URL.format(run=RUN): SRA}))
    monkeypatch.setattr(fetch, "convert_sra", _converter(9, _full))
    _, _, record = fetch_reads(RUN, tmp_path)
    manifest = fetch_manifest(tmp_path)
    assert json.loads((tmp_path / "fetch_manifest.json").read_text(encoding="utf-8")) == manifest
    assert manifest["kind"] == "alignair_assay.real.fetch_manifest" and manifest["sra_tools_image"] == SRA_TOOLS_IMAGE
    assert manifest["igenotyper"]["commit"] == fetch.IGENOTYPER_COMMIT
    entry = manifest["reads"][RUN]
    assert entry["path"] == f"airrseq_odp/{DONOR}/{RUN}" and entry["quality"] == "full" and entry["md5"] == record.md5
    assert manifest["quality"] == {RUN: "full"}
    assert RUN not in manifest["pending"]["reads"] and len(manifest["pending"]["reads"]) == 7
    assert sorted(manifest["pending"]["genotypes"]) == sorted(PILOT_DONORS)
    assert len(manifest["code_sha256"]) == 64


# --- the real data root ------------------------------------------------------------------------------------------

def _linked_root(tmp_path: Path) -> Path:
    """A temporary data root whose sources are links to the real root's directories: verified in place, while the
    records and the manifest are written to the temporary root."""
    real, root = Path(DATA), tmp_path / "root"
    (root / "genotypes").mkdir(parents=True)
    for entry in (real / "genotypes").iterdir():
        (root / "genotypes" / entry.name).symlink_to(entry)
    return root


@needs_data
def test_the_pins_are_the_data_roots_metadata():
    root = Path(DATA)
    with (root / "metadata/donor_runs.tsv").open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        rows = [dict(zip(header, line.rstrip("\n").split("\t"))) for line in handle]
    for donor, run in PILOT_DONORS.items():
        runs = [row for row in rows if row["donor"] == donor and row["data_type"] == "airrseq"]
        assert [row["run"] for row in runs] == [run]
        assert int(runs[0]["read_count"]) == fetch.READ_PAIRS[run]
        assert (root / "airrseq" / donor / run).is_dir()
    with (root / "metadata/pilot_files.tsv").open(encoding="utf-8") as handle:
        listed = {tuple(line.split("\t")[2:5]) for line in handle.read().splitlines()}
    pinned = {(ena_fastq_url(run, name).removeprefix("https://"), str(size), md5)
              for run in PILOT_DONORS.values() for name, size, md5 in fetch.ENA_FASTQ[run]}
    assert pinned == listed


@needs_data
def test_the_pilot_genotype_files_are_verified_in_place(tmp_path, monkeypatch):
    root = _linked_root(tmp_path)
    monkeypatch.setattr(fetch, "download", _no_download)
    for donor in PILOT_DONORS:
        for key, spec in genotype_specs(donor).items():
            if key == "gene_coords" and not (root / spec.path).exists():
                continue  # fetched by the network test
            record = fetch_file(spec, root)
            assert record.detail["reused"] is True and record.md5 == spec.md5


def test_the_cohort_files_are_pinned_https_urls_on_allowed_hosts():
    assert set(fetch.COHORT_FILES) == {"vdjbase_api", "ogrdb_rev9", "sd3"}
    for spec in fetch.COHORT_FILES.values():
        parsed = urllib.parse.urlsplit(spec.url)
        assert parsed.scheme == "https" and parsed.hostname in fetch.ALLOWED_HOSTS, spec.url
        assert len(spec.md5) == 32 and spec.bytes > 0 and spec.path.startswith("genotypes/")
    assert fetch.COHORT_FILES["vdjbase_api"].url == (
        "https://vdjbase.org/api/v1/genomic/all_subjects_genotype/Homo%20sapiens")
    assert fetch.COHORT_FILES["ogrdb_rev9"].url == "https://ogrdb.airr-community.org/api/germline/set/G00060/9/airr"
    assert fetch.COHORT_FILES["sd3"].url.endswith("/41467_2023_40070_MOESM4_ESM.xlsx")


def test_the_cohort_files_are_fetched_verified_and_reused(tmp_path, monkeypatch):
    payloads = {key: f"{key} bytes".encode() for key in fetch.COHORT_FILES}
    monkeypatch.setattr(fetch, "COHORT_FILES", {
        key: replace(spec, md5=_md5(payloads[key]), bytes=len(payloads[key]))
        for key, spec in fetch.COHORT_FILES.items()})
    calls: list = []
    served = {spec.url: payloads[key] for key, spec in fetch.COHORT_FILES.items()}
    monkeypatch.setattr(fetch, "download", _serving(served, calls))
    # a fresh data root fetches them from their pinned URLs, verified and listed
    records = fetch.cohort_files(tmp_path)
    assert set(records) == {"vdjbase_api", "ogrdb_rev9", "sd3"} and sorted(calls) == sorted(served)
    for key, record in records.items():
        spec = fetch.COHORT_FILES[key]
        assert record.path == tmp_path / spec.path and record.path.read_bytes() == payloads[key]
        assert record.md5 == spec.md5 and record.source_url == spec.url and record.kind == spec.kind
        assert spec.md5 in (record.path.parent / "MD5SUMS.txt").read_text()
    # reused once there, and recorded for the manifest
    assert all(record.detail["reused"] for record in fetch.cohort_files(tmp_path).values()) and len(calls) == 3
    assert fetch_manifest(tmp_path)["cohort"]["sd3"]["md5"] == fetch.COHORT_FILES["sd3"].md5
    # a file changed since is refused, and a download that is not the pinned file leaves nothing behind
    sd3 = tmp_path / fetch.COHORT_FILES["sd3"].path
    sd3.write_bytes(b"sd3 bytez")
    with pytest.raises(FetchError, match="md5"):
        fetch.cohort_files(tmp_path)
    sd3.unlink()
    served[fetch.COHORT_FILES["sd3"].url] = b"sd3 bytez"
    with pytest.raises(FetchError, match="md5"):
        fetch.cohort_files(tmp_path)
    assert not sd3.exists() and not sd3.with_name(sd3.name + ".part").exists()


@needs_data
def test_the_cohort_files_are_the_pinned_ones(tmp_path, monkeypatch):
    root = _linked_root(tmp_path)
    monkeypatch.setattr(fetch, "download", _no_download)
    records = fetch.cohort_files(root)
    assert {key: record.md5 for key, record in records.items()} == {
        key: spec.md5 for key, spec in fetch.COHORT_FILES.items()}
    assert all(record.detail["reused"] for record in records.values())


@needs_network
def test_the_cohort_files_download_from_their_pinned_urls(tmp_path, network):
    records = fetch.cohort_files(tmp_path)
    for key, record in records.items():
        assert record.md5 == fetch.COHORT_FILES[key].md5 and not record.detail["reused"]
        assert md5_file(record.path) == record.md5


@needs_data
def test_the_ena_fastqs_are_the_pinned_ones(tmp_path):
    root = Path(DATA)
    for donor, run in PILOT_DONORS.items():
        for name, size, md5 in fetch.ENA_FASTQ[run]:
            path = root / "airrseq" / donor / run / name
            assert path.stat().st_size == size and md5_file(path) == md5


@needs_data
@needs_network
def test_the_igenotyper_gene_coords_at_the_pinned_commit(network):
    records = fetch_genotype_files("W-97", DATA)
    coords = records["gene_coords"]
    assert coords.path == Path(DATA) / "genotypes/igenotyper/gene_coords.bed"
    lines = [line.split("\t") for line in coords.path.read_text(encoding="utf-8").splitlines()]
    igh = [fields[3][:4] for fields in lines if fields[0] == "igh"]
    assert len(igh) == 92 and {gene: igh.count(gene) for gene in set(igh)} == {"IGHV": 59, "IGHD": 27, "IGHJ": 6}
    assert max(int(fields[2]) for fields in lines if fields[0] == "igh") < 1_193_129  # the igh reference's length
    manifest = fetch_manifest(DATA)
    assert manifest["genotypes"]["W-97"]["gene_coords"]["md5"] == coords.md5


@needs_data
@needs_network
@needs_docker
def test_w97_odp_reads_are_full_quality(tmp_path, network):
    """The recon's deferred check: W-97's ODP object (34 MB), downloaded and converted in the pinned container into
    a temporary root, has full qualities (else the record says ODP is Lite and the reads are ENA's). The real root's
    ENA FASTQs of W-97 are linked in, so the conversion is compared with them."""
    root = tmp_path / "root"
    (root / "airrseq").mkdir(parents=True)
    (root / "airrseq" / "W-97").symlink_to(Path(DATA) / "airrseq" / "W-97")
    r1, r2, record = fetch_reads(PILOT_DONORS["W-97"], root)
    assert "reused" not in record.detail and record.detail["read_pairs"] == 103_049
    if record.detail["source"] == "odp":
        assert record.quality == "full" and quality_kind(r1) == quality_kind(r2) == "full"
        assert r1.parent == root / "airrseq_odp" / "W-97" / PILOT_DONORS["W-97"]
        assert record.detail["odp"]["bytes"] == 34_204_018 and "consistent" in record.detail["odp"]["vdb_validate"]
        assert record.detail["ena_agreement"] == {"records": fetch.QUALITY_RECORDS, "same": True}
        assert [record.detail["files"][p.name]["fastq_md5"] for p in (r1, r2)] == list(
            fetch.FASTQ_MD5[PILOT_DONORS["W-97"]])
    else:
        assert record.detail["odp"]["status"] in ("lite", "not found") and record.quality == "lite"
    assert fetch_manifest(root)["quality"] == {PILOT_DONORS["W-97"]: record.quality}
