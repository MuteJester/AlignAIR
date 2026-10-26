"""Verified downloads for the real-data pilot: each pilot run's reads, at full quality where AWS ODP has them, and
the donors' genotype files, all under a data root (``--data-root``), never in a run directory.

Everything fetched is pinned here (size, md5 or S3 ETag, read count, commit, image digest, and for the converted
reads the md5 of each mate's uncompressed FASTQ, which unlike its gzip does not depend on zlib), so a fresh data
root gets the same files, or the same reads, or a refusal. A file is verified before it is used: one already there
is reused only when it is the pinned file, and one downloaded is written to ``<name>.part`` and renamed only once
verified. Downloads are https only, from the pinned sources' hosts (``ALLOWED_HOSTS``; a redirect elsewhere is
refused at once), with a timeout and at most ``RETRIES`` retries; nothing downloaded is ever executed. Each source
has its own directory under the data root, with an ``MD5SUMS.txt`` this module keeps:

- ``airrseq/<donor>/<run>/<run>_{1,2}.fastq.gz``: ENA's FASTQs (``ENA_FASTQ``, ENA's md5s). ENA serves them with
  SRA-Lite qualities, every base ``?`` (Q30) or ``$`` (Q3), on 7 of the 8 pilot runs (all but SC-23).
- ``airrseq_odp/<donor>/<run>/``: the run's SRA object from NCBI's AWS Open Data mirror (``ODP_URL``; its size and
  S3 ETag pinned in ``ODP_OBJECT``) and the pair converted from it in the pinned sra-tools container
  (``SRA_TOOLS_IMAGE``): ``vdb-validate`` checks the object's internal md5s, ``fasterq-dump --split-files`` writes
  the mates with ENA's read names (``@<run>.<spot> <spot>/<mate>`` and a bare ``+``), and they are gzipped here,
  checked record by record to be paired. The container is the shared hardened run (``core.container``): as the
  current user, with no network, no capabilities and no privilege escalation, on a bind mount of the run's
  directory only (which must exist), named per run and process so that one that times out is killed; the image is
  checked by ``core.container.check_image``.
- ``genotypes/vdjbase_genomic_samples/<sample>.csv`` and ``.bam`` (VDJbase genomic study P25, CC0),
  ``genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json`` (OGRDB's IGH_VDJ germline set G00060 rev 8,
  CC0) and ``genotypes/igenotyper/gene_coords.bed`` (IGenotyper's gene coordinates on its ``igh`` reference, at
  commit ``IGENOTYPER_COMMIT``, the last to change the file; earlier commits differ by one base at some gene ends).
- The sensitivity analysis's cohort files (``COHORT_FILES``, ``cohort_files``):
  ``genotypes/vdjbase_genomic_api/genomic_all_subjects_genotype_Homo_sapiens.json`` (VDJbase's public API v1, every
  genomic subject's genotype), ``genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_9.json`` and
  ``genotypes/rodriguez2023_suppl/41467_2023_40070_MOESM4_ESM.xlsx`` (the paper's Supplementary Data, from the
  journal's static content host), each from its pinned URL with its pinned md5.

``fetch_reads`` prefers ODP: its reads are ENA's, base for base and name for name (``ena_agreement`` compares the
first ``QUALITY_RECORDS``), with the qualities ENA dropped. It falls back to the ENA FASTQs when the ODP object is
missing or converts to Lite qualities, and records why; the record's ``quality`` is the alphabet the reads have
(``quality_kind``), ``"full"`` only when both mates are. It stops with ``FetchError`` (naming the file) on any other
failure: Docker missing or another image digest (before any download), a download that is not the pinned file, a
read count that is not ENA's, or a file changed since its record. All eight pilot runs convert at full quality,
with ENA's read counts and ENA's first records; on W-97 (SRR23654538) all 103,049 pairs have ENA's names and bases,
with 27 quality symbols where ENA has two.

The records live in ``<data root>/fetch/`` (``reads/<run>.json``, ``genotypes/<donor>.json``) and
``fetch_manifest`` gathers them into ``<data root>/fetch_manifest.json``. A reads record is reused only while it is
current: its ``recipe_sha256`` is ``recipe_digest`` (the pinned image, the container commands, the read names, the
gzip level, the quality sample and the run's pins) and its image is the pinned one. A stale record is redone, the
ODP object being reused after its ETag check; a Lite fallback is reused only after ODP is tried once more.
"""
from __future__ import annotations

import gzip
import hashlib
import http.client
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from contextlib import ExitStack
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path

from alignair_assay.core import container
from alignair_assay.core.provenance import sha256_file

# --- the pilot ---------------------------------------------------------------------------------------------------

# Donor -> its AIRR-seq run (``metadata/donor_runs.tsv``, ``data_type`` airrseq). Each pilot donor has one such run:
# the pilot was drawn from the donors with one, so it is the run the pilot download put under ``airrseq/<donor>/``.
PILOT_DONORS: dict[str, str] = {
    "SC-23": "SRR23446586", "SC-4": "SRR23446584", "SC-11": "SRR23446589", "SC-6": "SRR23446583",
    "W-97": "SRR23654538", "W-89": "SRR23654540", "W-133": "SRR23654572", "W-131": "SRR23654574",
}
PILOT_SPLIT: dict[str, tuple[str, ...]] = {"dev": ("SC-23", "SC-11", "W-97", "W-133"),
                                           "test": ("SC-4", "SC-6", "W-89", "W-131")}
# Donor -> its VDJbase genomic sample (study P25), from the genomic subjects of VDJbase's API.
VDJBASE_SAMPLE: dict[str, str] = {
    "SC-23": "P25_I58_S1", "SC-4": "P25_I61_S1", "SC-11": "P25_I48_S1", "SC-6": "P25_I62_S1",
    "W-97": "P25_I151_S1", "W-89": "P25_I145_S1", "W-133": "P25_I93_S1", "W-131": "P25_I91_S1",
}
# Run -> its read pairs (ENA ``read_count``).
READ_PAIRS: dict[str, int] = {
    "SRR23446586": 164_664, "SRR23446584": 942_871, "SRR23446589": 989_139, "SRR23446583": 1_004_290,
    "SRR23654538": 103_049, "SRR23654540": 113_652, "SRR23654572": 127_136, "SRR23654574": 153_516,
}

# --- the reads ---------------------------------------------------------------------------------------------------

ODP_URL = "https://sra-pub-run-odp.s3.amazonaws.com/sra/{run}/{run}"
# Run -> its ODP object's size and S3 ETag (a multipart ETag of ``S3_PART_SIZE`` parts), as served on 2026-10-20.
ODP_OBJECT: dict[str, tuple[int, str]] = {
    "SRR23446586": (52_438_634, "0794d0683c283513b6bc1687d13c77fb-7"),
    "SRR23446584": (310_026_894, "cd927d94c04f71b4410ec6a05db39926-37"),
    "SRR23446589": (315_756_853, "cdcb6df6b20e94eb847b8f2b5b4dae44-38"),
    "SRR23446583": (327_806_693, "af97b937ac4af5b88d31646f693d09f0-40"),
    "SRR23654538": (34_204_018, "ed080b3e08ce1bc72b3b8b9fba7bb9dc-5"),
    "SRR23654540": (37_441_494, "210f3afc2bae7e2a4e668a30f288bf48-5"),
    "SRR23654572": (39_949_090, "17d45b70a0621c1474a1c0e50296d541-5"),
    "SRR23654574": (49_003_723, "cd1e33178674dabfa360ca168861ba73-6"),
}
# Run -> the md5s of its converted mates' uncompressed FASTQ streams (mate 1, mate 2): what ``sra_commands`` makes
# of the pinned object, independent of the zlib that gzips it. Checked on every conversion at full quality.
FASTQ_MD5: dict[str, tuple[str, str]] = {
    "SRR23446586": ("825acf77d2bf9855fa99d9680bea386e", "900dc2296c93d832c5e79585499127f2"),
    "SRR23446584": ("1b7c612daefaec6ebe28ed4dbd7b3cff", "481f62ebf1d2e480ad2cacb7b488bc6f"),
    "SRR23446589": ("ff73496fe0cd0625f87ff33c377f0c5c", "a67c282f38b31407f1325a9dbb279519"),
    "SRR23446583": ("d89067c9f85d5c0fedc926e6e52a9dcd", "0f2d156ede4181cf53447b4e051cb948"),
    "SRR23654538": ("f23817602d5755392360a035ea8cc5e1", "7642687f4c1575c513733c4cd9bfe35f"),
    "SRR23654540": ("40de9891d3c09f2e073306a7baf7935a", "3e01c6f0eca47a2a5b8a7b7061ccbb03"),
    "SRR23654572": ("035049578eec0c912c2f70276519d47a", "8c8e64cd4f5f69ae9918f79e0291e8a8"),
    "SRR23654574": ("9230266de4a700043545e737aaed4f26", "fbee1a2eedc70e8df6717fd8e64a3fe6"),
}
# The part size the ODP objects were uploaded in (the AWS CLI's default), which their multipart ETags depend on.
S3_PART_SIZE = 8 * 1024 * 1024
ENA_FASTQ_ROOT = "https://ftp.sra.ebi.ac.uk/vol1/fastq"
# Run -> its ENA FASTQs: (name, bytes, md5), mate 1 then mate 2 (ENA ``fastq_bytes`` and ``fastq_md5``).
ENA_FASTQ: dict[str, tuple[tuple[str, int, str], tuple[str, int, str]]] = {
    "SRR23446586": (("SRR23446586_1.fastq.gz", 20_326_042, "217dc9ea77b8746e6d665c7d60fac55a"),
                    ("SRR23446586_2.fastq.gz", 28_930_160, "5cf11e87e0c1dacae8360c5be34ed533")),
    "SRR23446584": (("SRR23446584_1.fastq.gz", 57_982_790, "731ab6a465ea4e6d59eb12f3404f2ae2"),
                    ("SRR23446584_2.fastq.gz", 55_714_861, "5b85df6153cd90cf27fd4c8d578e3a36")),
    "SRR23446589": (("SRR23446589_1.fastq.gz", 59_608_493, "4a7a731093cfab016dfac4fc2b2754eb"),
                    ("SRR23446589_2.fastq.gz", 60_464_373, "deb385623c122b9e6535396c27c42bb2")),
    "SRR23446583": (("SRR23446583_1.fastq.gz", 67_929_741, "181d29be14138e126c9580c26e72d469"),
                    ("SRR23446583_2.fastq.gz", 64_523_599, "c32fff523510f95fe6db6f041f7c7da2")),
    "SRR23654538": (("SRR23654538_1.fastq.gz", 3_688_736, "7ccca96b866affae5452a9d453a39d8b"),
                    ("SRR23654538_2.fastq.gz", 5_530_002, "6bf780cdc4ae2732699c92c287da8e64")),
    "SRR23654540": (("SRR23654540_1.fastq.gz", 3_656_456, "ee0a677f951f5665a8ee0bd8b9cf4337"),
                    ("SRR23654540_2.fastq.gz", 5_757_929, "fe8722a3f58c6858b1915eb6ff672424")),
    "SRR23654572": (("SRR23654572_1.fastq.gz", 4_766_279, "d8a36020b54231e99a029de6b1fae33a"),
                    ("SRR23654572_2.fastq.gz", 6_512_400, "4f6708b8cc423196e8362263d95ad2ef")),
    "SRR23654574": (("SRR23654574_1.fastq.gz", 4_886_487, "b2f3b42d8b33709cad141aaa3ede3dcd"),
                    ("SRR23654574_2.fastq.gz", 7_610_428, "7e4dd5c561c1e04906e2089a9b09a204")),
}
# ncbi/sra-tools 3.4.1, pushed 2026-03-30 (image sha256:cc7a4a445f5b...); fasterq-dump and vdb-validate 3.4.1.
SRA_TOOLS_IMAGE = ("ncbi/sra-tools:3.4.1"
                   "@sha256:4f048ba6754246e1ed7799a2c8ed469a38427f392e403ca08b93c4255104c9ce")
# fasterq-dump's read names, written as ENA writes them, so the two sources' FASTQs differ only in quality.
ENA_SEQ_DEFLINE = "@$ac.$si $sn/$ri"
# SRA-Lite's quality symbols: "?" (Q30) for a base that passed, "$" (Q3) for one that did not.
LITE_QUALITIES = frozenset("?$")
# The records ``quality_kind`` reads and ``ena_agreement`` compares.
QUALITY_RECORDS = 2000

# --- the genotype files ------------------------------------------------------------------------------------------

VDJBASE_DIR = "genotypes/vdjbase_genomic_samples"
VDJBASE_URL = "https://vdjbase.org/static/study_data/Genomic/samples/Human/IGH/P25/{sample}/{name}"
# Sample -> its CSV's and its contig BAM's (bytes, md5).
VDJBASE_FILES: dict[str, tuple[tuple[int, str], tuple[int, str]]] = {
    "P25_I58_S1": ((337_020, "cafd126ce6267568f6bd868277bb94a3"), (3_816_880, "9faed2f7bdc627cc71f20fdf4a5bcb41")),
    "P25_I61_S1": ((281_266, "5772bb95d64146c8a37cbdb0aaf87e03"), (3_862_806, "2221dab022bbeac941fa14f31f39d61f")),
    "P25_I48_S1": ((323_401, "d992213e7fb8228b4d8a952e391277f0"), (4_827_606, "7aa23721dde5531aa09204295b550913")),
    "P25_I62_S1": ((312_453, "cd36f8383c7e5a8f88230fd14648dbab"), (5_101_487, "4e859450e51c6cedad2cd88480ad98e9")),
    "P25_I151_S1": ((261_089, "16e3e5c498f76b5ca3c91edbceb7eb57"), (5_826_066, "3a11bea90553228d487ced00b9fc2945")),
    "P25_I145_S1": ((370_616, "875e4b6f84bd45ad41de6f35f5b39ef1"), (5_195_148, "495afca25a11cf6424bea24452b65adf")),
    "P25_I93_S1": ((253_128, "44bdf4f695ff2836d624c56d04e1b936"), (4_680_640, "48801776e304d9c50d58b124723c3ec1")),
    "P25_I91_S1": ((296_782, "57d99f5a1910d4f5667ec25c0e04b4af"), (3_935_340, "1ed9713a0bb578b84a707f7df9f2f62a")),
}
OGRDB_PATH = "genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_8.json"
OGRDB_URL = "https://ogrdb.airr-community.org/api/germline/set/G00060/8/airr"
OGRDB_FILE = (2_365_085, "296bf2ef309599827c1932537c45c18c")
IGENOTYPER_REPOSITORY = "oscarlr/IGenotyper"
IGENOTYPER_COMMIT = "40a00015f000adae477d25e62ec2e9e8dba9da98"  # 2022-10-11, the file as on master in 2026-09
GENE_COORDS_PATH = "genotypes/igenotyper/gene_coords.bed"
GENE_COORDS_URL = (f"https://raw.githubusercontent.com/{IGENOTYPER_REPOSITORY}/{IGENOTYPER_COMMIT}/"
                   "IGenotyper/data/gene_coords.bed")
GENE_COORDS_FILE = (13_006, "2ca651322855248cb053d76b0e981738")

# --- where things go, and how --------------------------------------------------------------------------------------

# The cohort-level files of the sensitivity analysis (``COHORT_FILES``, ``cohort_files``): VDJbase's public API v1,
# the journal's supplementary material (Rodriguez et al. 2023, Nat Commun 14:4419, doi:10.1038/s41467-023-40070-x).
VDJBASE_API_URL = "https://vdjbase.org/api/v1/genomic/all_subjects_genotype/Homo%20sapiens"
OGRDB_REV9_URL = "https://ogrdb.airr-community.org/api/germline/set/G00060/9/airr"
SD3_URL = ("https://static-content.springer.com/esm/art%3A10.1038%2Fs41467-023-40070-x/MediaObjects/"
           "41467_2023_40070_MOESM4_ESM.xlsx")

# The hosts of the pinned sources: a download, or a redirect, anywhere else is refused.
ALLOWED_HOSTS = frozenset({"sra-pub-run-odp.s3.amazonaws.com", "ftp.sra.ebi.ac.uk", "vdjbase.org",
                           "ogrdb.airr-community.org", "raw.githubusercontent.com", "static-content.springer.com"})
ENA_DIR = "airrseq"
ODP_DIR = "airrseq_odp"
RECORDS_DIR = "fetch"
MANIFEST = "fetch_manifest.json"
MANIFEST_KIND = "alignair_assay.real.fetch_manifest"
MANIFEST_VERSION = 1
MD5SUMS = "MD5SUMS.txt"
# The staging directory of a conversion, inside the run's directory (the container sees only that directory).
CONVERT_DIR = "convert"
TIMEOUT = 60.0  # seconds a download may wait on the server
RETRIES = 2  # retries of a failed download, after the first attempt
RETRY_WAIT = 5.0  # seconds before the first retry, doubled before the second
PULL_TIMEOUT = 1800.0
CONVERT_TIMEOUT = 3600.0
THREADS = 4
GZIP_LEVEL = 6
# Free space a conversion needs, in multiples of the ODP object's size (the object, the FASTQs and their gzip).
FREE_SPACE_FACTOR = 8
_CHUNK = 1 << 20
_USER_AGENT = "alignair-assay-fetch"
# This module's code, taken when it was imported: the process runs the code it imported, whatever the file becomes.
_CODE_SHA256 = sha256_file(__file__)


class FetchError(RuntimeError):
    """A download or a file that could not be verified, or a step that failed; the message names it."""


class NotFound(FetchError):
    """The server has no such object (HTTP 403 or 404; S3 answers 403 for a missing key)."""


class Refused(FetchError):
    """A download the policy forbids: a URL or a redirect that is not https, or not to a pinned source's host."""


@dataclass(frozen=True)
class FetchRecord:
    """One verified fetch: a file, or for a run's reads the directory of its pair."""

    # the file; for reads, the directory holding the pair (``detail["pair"]`` names its two files)
    path: Path
    # the file's md5; for reads, the md5 of the pair's ``MD5SUMS.txt`` lines (``<md5>  <name>``, mate 1 first)
    md5: str
    # the file's size; for reads, the pair's
    bytes: int
    source_url: str
    # "reads", "vdjbase_csv", "vdjbase_bam", "ogrdb" or "gene_coords"
    kind: str
    # for reads, the quality alphabet: "full" or "lite" (SRA-Lite); None for any other file
    quality: str | None = None
    # provenance: the source, the files, the pins and what was checked; not part of the record's identity
    detail: dict = field(default_factory=dict, compare=False, repr=False)

    def to_json(self, data_root) -> dict:
        """The record as JSON, its path relative to ``data_root``."""
        return {"path": Path(self.path).relative_to(data_root).as_posix(), "md5": self.md5, "bytes": self.bytes,
                "source_url": self.source_url, "kind": self.kind, "quality": self.quality,
                "detail": {key: value for key, value in self.detail.items() if key != "reused"}}

    @classmethod
    def from_json(cls, record: dict, data_root) -> "FetchRecord":
        return cls(path=Path(data_root) / record["path"], md5=record["md5"], bytes=record["bytes"],
                   source_url=record["source_url"], kind=record["kind"], quality=record["quality"],
                   detail=dict(record.get("detail") or {}))


@dataclass(frozen=True)
class FileSpec:
    """A pinned file: where it goes under the data root, where it comes from, and what it must be."""

    path: str  # POSIX, relative to the data root
    url: str
    md5: str
    bytes: int
    kind: str
    detail: dict = field(default_factory=dict, compare=False)


# --- the quality alphabet ----------------------------------------------------------------------------------------

def quality_kind(fastq_gz, n: int = QUALITY_RECORDS) -> str:
    """``"lite"`` when every quality symbol of the first ``n`` records of gzipped FASTQ ``fastq_gz`` is SRA-Lite's
    (``LITE_QUALITIES``), else ``"full"``. It only reads the file. ValueError for a file that is not gzip, is
    truncated, holds no record, or has a record that is not FASTQ (no ``@`` header or ``+`` separator, or a quality
    string whose length is not its sequence's)."""
    symbols: set[str] = set()
    count = 0
    try:
        with gzip.open(fastq_gz, "rt", encoding="ascii") as handle:
            while count < n:
                header = handle.readline()
                if not header:
                    break
                sequence, separator, quality = handle.readline(), handle.readline(), handle.readline()
                if not header.startswith("@") or not separator.startswith("+"):
                    raise ValueError(f"{fastq_gz}: record {count + 1} is not a FASTQ record")
                sequence, quality = sequence.rstrip("\n"), quality.rstrip("\n")
                if not quality or len(quality) != len(sequence):
                    raise ValueError(f"{fastq_gz}: record {count + 1} has a quality string of length {len(quality)} "
                                     f"for {len(sequence)} bases")
                symbols.update(quality)
                count += 1
    except (OSError, EOFError, UnicodeDecodeError) as error:  # gzip.BadGzipFile is an OSError
        raise ValueError(f"{fastq_gz} is not a readable gzip FASTQ (truncated, or not gzip): {error}") from error
    if not count:
        raise ValueError(f"{fastq_gz} holds no record")
    return "lite" if symbols <= LITE_QUALITIES else "full"


# --- checksums ---------------------------------------------------------------------------------------------------

def md5_file(path) -> str:
    """Hex md5 of the file's bytes."""
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_CHUNK), b""):
            digest.update(chunk)
    return digest.hexdigest()


def s3_etag(path, *, part_size: int | None = None, single: bool = False) -> str:
    """The ETag S3 gives an object of the file's bytes: its md5 when uploaded whole (``single``), else, uploaded in
    parts of ``part_size`` bytes (default ``S3_PART_SIZE``), the md5 of the parts' md5s and ``-<parts>``."""
    if single:
        return md5_file(path)
    size = part_size or S3_PART_SIZE
    parts = []
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(size), b""):
            parts.append(hashlib.md5(chunk).digest())
    return f"{hashlib.md5(b''.join(parts)).hexdigest()}-{len(parts)}"


def _listing_md5(files: dict[str, dict], names) -> str:
    """The md5 of the ``MD5SUMS.txt`` lines of ``names`` (``<md5>  <name>``), in that order."""
    return hashlib.md5("".join(f"{files[name]['md5']}  {name}\n" for name in names).encode()).hexdigest()


def _listed_md5(path: Path) -> str | None:
    """The md5 the ``MD5SUMS.txt`` beside ``path`` lists for it; None when it lists none."""
    listing = path.parent / MD5SUMS
    if not listing.is_file():
        return None
    for line in listing.read_text(encoding="utf-8").splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2 and parts[1].lstrip("*").strip() == path.name:
            return parts[0].lower()
    return None


def _list_md5(path: Path, md5: str) -> None:
    """List ``path``'s md5 in the ``MD5SUMS.txt`` beside it, replacing a line it already has."""
    listing = path.parent / MD5SUMS
    lines = listing.read_text(encoding="utf-8").splitlines() if listing.is_file() else []
    kept = [line for line in lines
            if len(line.split(maxsplit=1)) != 2 or line.split(maxsplit=1)[1].lstrip("*").strip() != path.name]
    _write_text(listing, "".join(f"{line}\n" for line in [*kept, f"{md5}  {path.name}"]))


def _verify(path: Path, md5: str, size: int, what: str) -> None:
    """FetchError unless ``path`` is ``size`` bytes with md5 ``md5``."""
    found = path.stat().st_size
    if found != size:
        raise FetchError(f"{what} ({path}) is {found} bytes, not the pinned {size} bytes")
    digest = md5_file(path)
    if digest != md5:
        raise FetchError(f"{what} ({path}) has md5 {digest}, not the pinned {md5}")


def _verify_object(path: Path, size: int, etag: str, what: str) -> None:
    """FetchError unless ``path`` is ``size`` bytes with S3 ETag ``etag``."""
    found = path.stat().st_size
    if found != size:
        raise FetchError(f"{what} ({path}) is {found} bytes, not the pinned {size} bytes")
    digest = s3_etag(path, single="-" not in etag)
    if digest != etag:
        raise FetchError(f"{what} ({path}) has S3 ETag {digest}, not the pinned {etag}")


# --- downloads ---------------------------------------------------------------------------------------------------

def download(url: str, dest, *, timeout: float | None = None, retries: int | None = None) -> dict:
    """Stream https ``url`` into ``dest`` (overwritten), waiting at most ``timeout`` seconds (``TIMEOUT``) on the
    server and retrying a failed attempt at most ``retries`` times (``RETRIES``). ``{"url", "bytes", "md5",
    "headers"}`` (Content-Length, ETag, Last-Modified). Refused, before any request, for a URL that is not https
    or not on an ``ALLOWED_HOSTS`` host, and, before anything is written, for a redirect off them; NotFound for
    HTTP 403 or 404. Neither is retried. FetchError when every attempt failed, or for another client error."""
    _check_allowed(url, url)
    timeout = TIMEOUT if timeout is None else timeout
    retries = RETRIES if retries is None else retries
    last: Exception | None = None
    for attempt in range(retries + 1):
        if attempt:
            time.sleep(RETRY_WAIT * 2 ** (attempt - 1))
        try:
            return _download_once(url, Path(dest), timeout)
        except (NotFound, Refused):
            raise
        except urllib.error.HTTPError as error:
            if error.code in (403, 404):
                raise NotFound(f"{url}: HTTP {error.code} ({error.reason})") from error
            if error.code < 500 and error.code != 429:
                raise FetchError(f"{url}: HTTP {error.code} ({error.reason})") from error
            last = error
        except (FetchError, urllib.error.URLError, OSError, http.client.HTTPException) as error:
            last = error  # a timeout is an OSError, a short body a FetchError
    raise FetchError(f"{url}: {retries + 1} attempts failed; the last: {last}")


def _check_allowed(url: str, requested: str) -> None:
    """Refused unless ``url`` is https on one of ``ALLOWED_HOSTS``; ``requested`` is the URL asked for, which differs
    from ``url`` after a redirect."""
    parts = urllib.parse.urlsplit(url)
    redirect = f"{requested} was redirected to {url}: " if url != requested else f"{url}: "
    if parts.scheme != "https":
        raise Refused(f"{redirect}only https downloads are made")
    if parts.hostname not in ALLOWED_HOSTS:
        raise Refused(f"{redirect}host {parts.hostname} is not a pinned source ({', '.join(sorted(ALLOWED_HOSTS))})")


def _download_once(url: str, dest: Path, timeout: float) -> dict:
    request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
    digest, size = hashlib.md5(), 0
    with urllib.request.urlopen(request, timeout=timeout) as response:
        _check_allowed(response.geturl(), url)
        headers = {key: response.headers.get(key) for key in ("Content-Length", "ETag", "Last-Modified")}
        with dest.open("wb") as out:
            for chunk in iter(lambda: response.read(_CHUNK), b""):
                out.write(chunk)
                digest.update(chunk)
                size += len(chunk)
    if headers["Content-Length"] is not None and int(headers["Content-Length"]) != size:
        raise FetchError(f"{url}: {size} of {headers['Content-Length']} bytes arrived")
    return {"url": url, "bytes": size, "md5": digest.hexdigest(), "headers": headers}


def fetch_file(spec: FileSpec, data_root) -> FetchRecord:
    """The pinned file ``spec`` under ``data_root``, verified by size and md5: reused when already there (``detail``
    ``reused``), else downloaded to ``<name>.part``, verified, renamed and listed in its directory's
    ``MD5SUMS.txt``. FetchError, leaving nothing behind, for a download that is not the pinned file; FetchError,
    leaving it as it is, for a file already there that is not, or one its ``MD5SUMS.txt`` lists with another md5."""
    path = Path(data_root) / spec.path
    listed = _listed_md5(path)
    if listed is not None and listed != spec.md5:
        raise FetchError(f"{path}: its {MD5SUMS} lists md5 {listed}, not the pinned {spec.md5}")
    reused = path.exists()
    if reused:
        _verify(path, spec.md5, spec.bytes, "the file already there")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        try:
            download(spec.url, part)
            _verify(part, spec.md5, spec.bytes, f"the download of {spec.url}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(path)
        _list_md5(path, spec.md5)
    return FetchRecord(path=path, md5=spec.md5, bytes=spec.bytes, source_url=spec.url, kind=spec.kind,
                       detail={**spec.detail, "reused": reused})


# --- the genotype files ------------------------------------------------------------------------------------------

def genotype_specs(donor: str) -> dict[str, FileSpec]:
    """The pinned genotype files of pilot donor ``donor``: ``vdjbase_csv``, ``vdjbase_bam``, ``ogrdb`` and
    ``gene_coords`` (the last two shared by every donor). ValueError for a donor outside the pilot."""
    sample = _sample(donor)
    (csv_bytes, csv_md5), (bam_bytes, bam_md5) = VDJBASE_FILES[sample]
    vdjbase = {"sample": sample, "donor": donor, "study": "P25"}
    return {
        "vdjbase_csv": FileSpec(f"{VDJBASE_DIR}/{sample}.csv", VDJBASE_URL.format(sample=sample, name=f"{sample}.csv"),
                                csv_md5, csv_bytes, "vdjbase_csv", vdjbase),
        "vdjbase_bam": FileSpec(f"{VDJBASE_DIR}/{sample}.bam", VDJBASE_URL.format(sample=sample, name=f"{sample}.bam"),
                                bam_md5, bam_bytes, "vdjbase_bam", vdjbase),
        "ogrdb": FileSpec(OGRDB_PATH, OGRDB_URL, OGRDB_FILE[1], OGRDB_FILE[0], "ogrdb",
                          {"germline_set_id": "G00060", "release_version": 8}),
        "gene_coords": FileSpec(GENE_COORDS_PATH, GENE_COORDS_URL, GENE_COORDS_FILE[1], GENE_COORDS_FILE[0],
                                "gene_coords", {"repository": IGENOTYPER_REPOSITORY, "commit": IGENOTYPER_COMMIT,
                                                "file": "IGenotyper/data/gene_coords.bed"}),
    }


def fetch_genotype_files(donor: str, data_root) -> dict[str, FetchRecord]:
    """Pilot donor ``donor``'s genotype files (``genotype_specs``), each verified by md5 and reused when already
    there, keyed as there; their records are written to ``<data root>/fetch/genotypes/<donor>.json``. ValueError
    for a donor outside the pilot; FetchError as ``fetch_file`` raises it."""
    root = Path(data_root)
    records = {key: fetch_file(spec, root) for key, spec in genotype_specs(donor).items()}
    _write_json(root / RECORDS_DIR / "genotypes" / f"{donor}.json",
                {key: record.to_json(root) for key, record in records.items()})
    return records


# The cohort-level files the sensitivity analysis reads, as served on 2026-10-21 (``cohort_files``): VDJbase's genomic
# genotypes of every subject (its public API v1, MiAIRR GenotypeSets, CC0), OGRDB's IGH germline set rev 9 (the
# release those genotypes name their documented alleles in, CC0) and Rodriguez et al. 2023's supplementary workbook
# (its Supplementary Data 3 lists the paper's novel alleles, CC BY 4.0). The API's JSON is served live: a VDJbase
# update would change it, and the md5 then refuses it until the pin is moved.
COHORT_FILES: dict[str, FileSpec] = {
    "vdjbase_api": FileSpec("genotypes/vdjbase_genomic_api/genomic_all_subjects_genotype_Homo_sapiens.json",
                            VDJBASE_API_URL, "29c3ffae0478e5229107643ff9041840", 6_019_375, "vdjbase_api",
                            {"source": "VDJbase API v1 genomic/all_subjects_genotype/Homo sapiens", "licence": "CC0"}),
    "ogrdb_rev9": FileSpec("genotypes/ogrdb_human_IGH_VDJ/Homo_sapiens_IGH_VDJ_rev_9.json", OGRDB_REV9_URL,
                           "1c30b9ab4d0b17bd8c8497c740e56714", 2_369_576, "ogrdb",
                           {"germline_set_id": "G00060", "release_version": 9, "licence": "CC0"}),
    "sd3": FileSpec("genotypes/rodriguez2023_suppl/41467_2023_40070_MOESM4_ESM.xlsx", SD3_URL,
                    "8cfa14cb96358873623aa586369e5a94", 204_909, "sd3",
                    {"source": "Rodriguez et al. 2023, Nat Commun 14:4419, Supplementary Data 1-7",
                     "licence": "CC BY 4.0"}),
}


def cohort_files(data_root) -> dict[str, FetchRecord]:
    """The cohort files (``COHORT_FILES``) under ``data_root``, each fetched from its pinned URL when it is missing
    and verified by size and md5 (``fetch_file``: reused when already there, a download renamed only once verified),
    keyed as there; their records are written to ``<data root>/fetch/cohort.json``. FetchError as ``fetch_file``
    raises it."""
    root = Path(data_root)
    records = {key: fetch_file(spec, root) for key, spec in COHORT_FILES.items()}
    _write_json(root / RECORDS_DIR / "cohort.json", {key: record.to_json(root) for key, record in records.items()})
    return records


# --- the reads ---------------------------------------------------------------------------------------------------

def donor_of(run: str) -> str:
    """The pilot donor of AIRR-seq run ``run``. ValueError for a run outside the pilot."""
    for donor, pilot_run in PILOT_DONORS.items():
        if pilot_run == run:
            return donor
    raise ValueError(f"{run} is not a pilot run: the pins cover {', '.join(PILOT_DONORS.values())}")


def ena_fastq_url(run: str, name: str) -> str:
    """The https URL of ENA FASTQ ``name`` of ``run``: ``<root>/<first 6 characters>/<subdirectory>/<run>/<name>``,
    the subdirectory being the accession's digits after the sixth, zero-padded to three (none for six digits)."""
    digits = run[3:]
    subdirectory = f"/{digits[6:].zfill(3)}" if len(digits) > 6 else ""
    return f"{ENA_FASTQ_ROOT}/{run[:6]}{subdirectory}/{run}/{name}"


def ena_specs(run: str) -> tuple[FileSpec, FileSpec]:
    """The pinned ENA FASTQs of pilot run ``run``, mate 1 then mate 2."""
    donor = donor_of(run)
    return tuple(FileSpec(f"{ENA_DIR}/{donor}/{run}/{name}", ena_fastq_url(run, name), md5, size, "ena_fastq",
                          {"run": run, "mate": mate})
                 for mate, (name, size, md5) in enumerate(ENA_FASTQ[run], start=1))


def check_image(image: str | None = None) -> dict:
    """The pinned sra-tools image (``SRA_TOOLS_IMAGE``), pulled by digest when it is not local
    (``core.container.check_image``): its ``image`` reference, ``id`` and ``repo_digests``. FetchError when Docker is
    not available, the image cannot be pulled or inspected, or its RepoDigests lack the pinned digest."""
    try:
        return container.check_image(image or SRA_TOOLS_IMAGE, pull_timeout=PULL_TIMEOUT)
    except container.ContainerError as error:
        raise FetchError(f"{error} (the reads are converted in the pinned sra-tools container)") from error


# The placeholders of the container commands' run-specific parts (``_command_templates``).
_RUN_DIR, _RUN, _USER, _NAME, _THREADS = "<run_dir>", "<run>", "<user>", "<name>", "<threads>"


def _commands(run_dir: str, run: str, name: str, threads: str) -> list[list[str]]:
    """The two container runs of a conversion (``core.container.hardened_run``), on ``run_dir`` alone mounted
    writable at ``/data``: ``vdb-validate`` of the object, then ``fasterq-dump --split-files`` with ENA's read
    names."""
    mounts = [container.Mount(run_dir, "/data", read_only=False)]

    def run_in(args: list[str]) -> list[str]:
        return container.hardened_run(SRA_TOOLS_IMAGE, args, name=name, mounts=mounts, workdir="/data",
                                      env={"HOME": "/tmp"})

    return [run_in(["vdb-validate", f"/data/{run}"]),
            run_in(["fasterq-dump", "--split-files", "--threads", threads, "--temp", f"/data/{CONVERT_DIR}/tmp",
                    "--outdir", f"/data/{CONVERT_DIR}", "--seq-defline", ENA_SEQ_DEFLINE, "--qual-defline", "+",
                    f"/data/{run}"])]


def _command_templates() -> list[list[str]]:
    """The container commands of a conversion with their run-specific parts as placeholders (the run directory,
    the run, the user, the container's name and the threads): what ``recipe_digest`` digests."""
    templates = _commands(f"/{_RUN_DIR}", _RUN, _NAME, _THREADS)
    for argv in templates:
        argv[argv.index("--user") + 1] = _USER
    return templates


def container_name(run: str) -> str:
    """The name of this process's conversion containers for ``run``, so a timed-out one can be killed
    (``core.container.container_name``)."""
    return container.container_name("fetch", run)


def sra_commands(run_dir, run: str, *, threads: int | None = None) -> list[list[str]]:
    """The two container runs that turn ``<run_dir>/<run>`` into the pair in ``<run_dir>/convert/``:
    ``vdb-validate`` of the object, then ``fasterq-dump --split-files`` with ENA's read names. Each runs the pinned
    image as the current user, named ``container_name(run)``, with no network, no capabilities and no privilege
    escalation, on ``run_dir`` alone, bound at ``/data`` (``core.container.hardened_run``)."""
    return _commands(str(Path(run_dir).absolute()), run, container_name(run), str(threads or THREADS))


def recipe_digest(run: str) -> str:
    """SHA-256 of what makes pilot run ``run``'s reads, so a reads record is reused only while it is current: the
    pinned image (its digest included), the container commands with their run-specific parts as placeholders, the
    read names, ``GZIP_LEVEL`` and ``QUALITY_RECORDS``, and the run's pins (its ODP object's ETag and size, its read
    pairs, the ENA md5s and the converted FASTQs' md5s). Not the code's digest: an edit elsewhere in the module does
    not redo a conversion."""
    size, etag = ODP_OBJECT[run]
    recipe = {"image": SRA_TOOLS_IMAGE, "commands": _command_templates(), "seq_defline": ENA_SEQ_DEFLINE,
              "gzip_level": GZIP_LEVEL, "quality_records": QUALITY_RECORDS, "odp": {"etag": etag, "bytes": size},
              "read_pairs": READ_PAIRS[run], "ena_md5": [md5 for _, _, md5 in ENA_FASTQ[run]],
              "fastq_md5": list(FASTQ_MD5[run])}
    return hashlib.sha256(json.dumps(recipe, sort_keys=True).encode()).hexdigest()


def convert_sra(sra_path, out_dir, *, threads: int | None = None) -> dict:
    """Validate SRA object ``sra_path`` and write its mates as FASTQs into ``out_dir``, which must be the
    ``convert`` directory beside it (``sra_commands``). ``{"commands", "vdb_validate", "fasterq_dump"}``: the
    commands and the tools' last report lines. FetchError when either step fails, or does not finish within
    ``CONVERT_TIMEOUT``: its container is then killed by name."""
    sra_path = Path(sra_path).absolute()
    if Path(out_dir).absolute() != sra_path.parent / CONVERT_DIR:
        raise ValueError(f"{out_dir} is not {sra_path.parent / CONVERT_DIR}, the only directory the container sees")
    commands = sra_commands(sra_path.parent, sra_path.name, threads=threads)
    name = container_name(sra_path.name)
    reports = []
    for argv in commands:
        tool = argv[argv.index(SRA_TOOLS_IMAGE) + 1]
        try:  # on a timeout, its container is killed by name (core.container.run_container)
            completed = container.run_container(argv, name=name, timeout=CONVERT_TIMEOUT, what=f"{tool} of {sra_path}")
        except container.ContainerError as error:
            raise FetchError(str(error)) from error
        reports.append((completed.stdout + completed.stderr).strip())
    return {"commands": commands, "vdb_validate": (reports[0].splitlines() or [""])[-1],
            "fasterq_dump": " / ".join(line.strip() for line in reports[1].splitlines())}


def fetch_reads(run: str, data_root, *, threads: int | None = None) -> tuple[Path, Path, FetchRecord]:
    """Pilot run ``run``'s pair of gzipped FASTQs (mate 1, mate 2) and their record (see the module docstring).

    Its record (``<data root>/fetch/reads/<run>.json``) is reused while it is current, its ``recipe_sha256`` being
    ``recipe_digest(run)`` and its image's RepoDigests holding the pinned digest, and its files are unchanged; a
    Lite fallback is reused only after ODP is tried once more, and replaced if ODP now converts at full quality.
    A stale record is redone (``detail["replaced"]`` says why), the ODP object being reused after its ETag check.
    Otherwise the reads come from ODP at full quality, else ENA's FASTQs.

    ValueError for a run outside the pilot; FetchError, before any download, when Docker or the pinned image is not
    available; FetchError for a file changed since its current record, a download that is not the pinned object or
    file, or a conversion that is not the run's pinned reads."""
    donor = donor_of(run)
    root = Path(data_root)
    record_path = root / RECORDS_DIR / "reads" / f"{run}.json"
    recipe = recipe_digest(run)
    replaced = None
    if record_path.is_file():
        record = FetchRecord.from_json(_read_json(record_path), root)
        stale = _staleness(record, recipe)
        if stale is None:
            _check_unchanged(record, record_path)
            if record.quality != "lite":
                return _reused(record)
            image = check_image()
            odp = _odp_reads(run, donor, root, image, threads)
            if odp["status"] != "full":
                return _reused(record, odp_retry={"status": odp["status"], "reason": odp["reason"]})
            stale = "it was a Lite fallback, and ODP now converts at full quality"
            replaced = {"reason": stale, "md5": record.md5, "quality": record.quality,
                        "source": record.detail.get("source")}
            return _record_reads(run, donor, root, record_path, recipe, image, odp, replaced)
        replaced = {"reason": stale, "md5": record.md5, "quality": record.quality,
                    "source": record.detail.get("source")}
    image = check_image()
    odp = _odp_reads(run, donor, root, image, threads)
    return _record_reads(run, donor, root, record_path, recipe, image, odp, replaced)


def _staleness(record: FetchRecord, recipe: str) -> str | None:
    """Why a reads record is not current (None when it is): another recipe, or an image without the pinned digest."""
    found = record.detail.get("recipe_sha256")
    if found != recipe:
        return f"its recipe {found[:12] if found else '(none recorded)'} is not the current {recipe[:12]}"
    pinned = container.pinned_repo_digest(SRA_TOOLS_IMAGE)
    digests = (record.detail.get("image") or {}).get("repo_digests") or []
    if pinned not in digests:
        return f"its image's RepoDigests {digests} lack the pinned {pinned}"
    return None


def _reused(record: FetchRecord, **extra) -> tuple[Path, Path, FetchRecord]:
    first, second = (record.path / name for name in record.detail["pair"])
    return first, second, replace(record, detail={**record.detail, "reused": True, **extra})


def _record_reads(run: str, donor: str, root: Path, record_path: Path, recipe: str, image: dict, odp: dict,
                  replaced: dict | None) -> tuple[Path, Path, FetchRecord]:
    """The reads record of an ODP attempt ``odp``: ODP's pair at full quality, else ENA's FASTQs; written."""
    if odp["status"] == "full":
        run_dir = root / ODP_DIR / donor / run
        files, pair = odp.pop("files"), odp.pop("pair")
        detail = {"source": "odp", "ena_agreement": odp.pop("ena_agreement")}
        record = FetchRecord(path=run_dir, md5=_listing_md5(files, pair), bytes=sum(files[n]["bytes"] for n in pair),
                             source_url=odp["url"], kind="reads", quality="full")
    else:
        specs = ena_specs(run)
        fetched = [fetch_file(spec, root) for spec in specs]
        pair = [item.path.name for item in fetched]
        files = {item.path.name: {"md5": item.md5, "bytes": item.bytes, "quality": _quality(item.path)}
                 for item in fetched}
        quality = "full" if all(files[name]["quality"] == "full" for name in pair) else "lite"
        detail = {"source": "ena", "fallback_reason": odp["reason"]}
        record = FetchRecord(path=fetched[0].path.parent, md5=_listing_md5(files, pair),
                             bytes=sum(files[n]["bytes"] for n in pair), source_url=ena_fastq_url(run, ""),
                             kind="reads", quality=quality)
    detail.update(run=run, donor=donor, pair=pair, files=files, read_pairs=READ_PAIRS[run], odp=odp,
                  image=image, recipe_sha256=recipe, code_sha256=_CODE_SHA256,
                  fetched_at=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    if replaced:
        detail["replaced"] = replaced
    record = replace(record, detail=detail)
    _write_json(record_path, record.to_json(root))
    return record.path / pair[0], record.path / pair[1], record


def _odp_reads(run: str, donor: str, root: Path, image: dict, threads: int | None) -> dict:
    """The run's ODP object downloaded (or reused) and verified, and converted: ``status`` ``"full"`` with the
    pair moved into its directory (``files``, ``pair``, ``ena_agreement``), ``"lite"`` with the conversion
    discarded, or ``"not found"``; with the object's pins, its md5 and the container's reports."""
    url = ODP_URL.format(run=run)
    size, etag = ODP_OBJECT[run]
    run_dir = root / ODP_DIR / donor / run
    sra = run_dir / run
    odp = {"url": url, "bytes": size, "etag": etag}
    _check_free_space(root, size)  # for the conversion too, when the object is already there
    if sra.exists():
        _verify_object(sra, size, etag, "the ODP object already there")
    else:
        run_dir.mkdir(parents=True, exist_ok=True)
        part = sra.with_name(sra.name + ".part")
        try:
            download(url, part)
            _verify_object(part, size, etag, f"the download of {url}")
        except NotFound as error:
            part.unlink(missing_ok=True)
            if not any(run_dir.iterdir()):
                run_dir.rmdir()  # nothing of this source was fetched
            return {**odp, "status": "not found", "reason": f"the ODP object is missing: {error}"}
        except BaseException:
            part.unlink(missing_ok=True)
            raise
        part.replace(sra)
    odp["md5"] = md5_file(sra)
    _list_md5(sra, odp["md5"])
    staging = run_dir / CONVERT_DIR
    shutil.rmtree(staging, ignore_errors=True)
    try:
        converted = convert_sra(sra, staging, threads=threads)
        odp.update(commands=converted["commands"], vdb_validate=converted["vdb_validate"],
                   fasterq_dump=converted.get("fasterq_dump"))
        pair = [f"{run}_1.fastq.gz", f"{run}_2.fastq.gz"]
        written = sorted(path.name for path in staging.glob("*.fastq"))
        if written != [name.removesuffix(".gz") for name in pair]:
            raise FetchError(f"fasterq-dump wrote {written or 'nothing'} for {run}, not the pair "
                             f"{[name.removesuffix('.gz') for name in pair]}")
        mates = _gzip_pair([staging / name.removesuffix(".gz") for name in pair], [staging / name for name in pair])
        files = {}
        for name, mate in zip(pair, mates):
            if mate["records"] != READ_PAIRS[run]:
                raise FetchError(f"{name} from ODP holds {mate['records']} records, not the {READ_PAIRS[run]} read "
                                 f"pairs ENA lists for {run}")
            files[name] = {"md5": md5_file(staging / name), "bytes": (staging / name).stat().st_size,
                           "records": mate["records"], "fastq_md5": mate["fastq_md5"],
                           "quality": _quality(staging / name)}
        kinds = {name: files[name]["quality"] for name in pair}
        if set(kinds.values()) != {"full"}:
            return {**odp, "status": "lite",
                    "reason": f"the ODP qualities are SRA-Lite ({', '.join(f'{n}: {k}' for n, k in kinds.items())})"}
        for name, pinned in zip(pair, FASTQ_MD5[run]):
            if files[name]["fastq_md5"] != pinned:
                raise FetchError(f"{name} from ODP is not the pinned reads of {run}: its FASTQ has md5 "
                                 f"{files[name]['fastq_md5']}, not the pinned {pinned}")
        agreement = _ena_agreement(run, donor, root, [staging / name for name in pair])
        for name in pair:
            (staging / name).replace(run_dir / name)
            _list_md5(run_dir / name, files[name]["md5"])
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {**odp, "status": "full", "files": files, "pair": pair, "ena_agreement": agreement}


def _ena_agreement(run: str, donor: str, root: Path, pair: list[Path]) -> dict | None:
    """``{"records", "same": True}`` when the first ``QUALITY_RECORDS`` records of the ODP mates have the read names
    and bases of the ENA FASTQs; None when those are not under the data root. FetchError when they differ."""
    ena = [root / ENA_DIR / donor / run / name for name, _, _ in ENA_FASTQ[run]]
    if not all(path.is_file() for path in ena):
        return None
    compared = 0
    for odp_path, ena_path in zip(pair, ena):
        with gzip.open(odp_path, "rt") as odp_handle, gzip.open(ena_path, "rt") as ena_handle:
            for index in range(QUALITY_RECORDS):
                ours = [odp_handle.readline() for _ in range(4)]
                theirs = [ena_handle.readline() for _ in range(4)]
                if not ours[0] and not theirs[0]:
                    break
                if ours[:2] != theirs[:2]:
                    raise FetchError(f"{odp_path.name} from ODP is not the ENA reads of {run}: record {index + 1} "
                                     f"is {ours[0].strip()!r}, ENA's {theirs[0].strip()!r} (or their bases differ)")
                compared = max(compared, index + 1)
    return {"records": compared, "same": True}


def _gzip_pair(sources: list[Path], dests: list[Path]) -> list[dict]:
    """Gzip the two mates' FASTQs ``sources`` into ``dests`` (no name or time in the gzip header), in step, checking
    over the whole files that record by record they are FASTQ records of the same spot (the same first word of the
    header line); remove the sources. Per mate, ``{"records", "fastq_md5"}``, the md5 of its uncompressed stream,
    which unlike the gzip does not depend on zlib. FetchError for mates that are not paired, or not FASTQ."""
    digests = [hashlib.md5(), hashlib.md5()]
    buffers: list[list[bytes]] = [[], []]
    lines = 0
    with ExitStack() as stack:
        inputs = [stack.enter_context(path.open("rb")) for path in sources]
        outputs = [stack.enter_context(gzip.GzipFile(filename="", mode="wb", compresslevel=GZIP_LEVEL, mtime=0,
                                                     fileobj=stack.enter_context(path.open("wb"))))
                   for path in dests]

        def flush():
            for buffer, output, digest in zip(buffers, outputs, digests):
                data = b"".join(buffer)
                output.write(data)
                digest.update(data)
                buffer.clear()

        for first, second in zip_longest(*inputs):
            if first is None or second is None:
                longer = 2 if first is None else 1
                raise FetchError(f"{sources[0].name} and {sources[1].name} are not paired: mate {longer} has more "
                                 f"lines (from line {lines + 1})")
            phase = lines % 4
            if phase == 0:
                if first[:1] != b"@" or second[:1] != b"@":
                    raise FetchError(f"record {lines // 4 + 1} of {sources[0].name} or {sources[1].name} is not a "
                                     "FASTQ record (no @ header)")
                spot, other = first.split(None, 1)[0], second.split(None, 1)[0]
                if spot != other:
                    raise FetchError(f"record {lines // 4 + 1}: mate 1 is {spot.decode(errors='replace')!r}, mate 2 "
                                     f"{other.decode(errors='replace')!r}: the mates are not paired")
            elif phase == 2 and (first[:1] != b"+" or second[:1] != b"+"):
                raise FetchError(f"record {lines // 4 + 1} of {sources[0].name} or {sources[1].name} is not a "
                                 "FASTQ record (no + separator)")
            buffers[0].append(first)
            buffers[1].append(second)
            lines += 1
            if len(buffers[0]) >= 1 << 16:
                flush()
        flush()
    if lines % 4:
        raise FetchError(f"{sources[0].name} and {sources[1].name} have {lines} lines, not whole 4-line records")
    for path in sources:
        path.unlink()
    return [{"records": lines // 4, "fastq_md5": digest.hexdigest()} for digest in digests]


def _quality(path: Path) -> str:
    """``quality_kind`` of a FASTQ being fetched, its ValueError a FetchError: the fetch stops with the reason."""
    try:
        return quality_kind(path)
    except ValueError as error:
        raise FetchError(f"the quality check of {path} failed: {error}") from error


def _check_free_space(root: Path, size: int) -> None:
    """FetchError when the data root's file system has less than ``FREE_SPACE_FACTOR`` times ``size`` free."""
    existing = next(path for path in (root, *root.parents) if path.exists())
    free = shutil.disk_usage(existing).free
    if free < FREE_SPACE_FACTOR * size:
        raise FetchError(f"{existing} has {free / 1e9:.2f} GB free; converting an ODP object of {size / 1e6:.0f} MB "
                         f"needs {FREE_SPACE_FACTOR * size / 1e9:.2f} GB")


def _check_unchanged(record: FetchRecord, record_path: Path) -> None:
    """FetchError unless each file of a reads record still has its recorded size and md5."""
    for name, recorded in record.detail["files"].items():
        path = record.path / name
        if not path.is_file():
            problem = "it is missing"
        elif path.stat().st_size != recorded["bytes"]:
            problem = f"{path.stat().st_size} bytes, recorded {recorded['bytes']}"
        elif md5_file(path) != recorded["md5"]:
            problem = f"md5 {md5_file(path)}, recorded {recorded['md5']}"
        else:
            continue
        raise FetchError(f"{path} has changed since it was fetched ({problem}); move it and its record "
                         f"{record_path} away to fetch it again")


def _sample(donor: str) -> str:
    try:
        return VDJBASE_SAMPLE[donor]
    except KeyError:
        raise ValueError(f"{donor} is not a pilot donor: the pins cover {', '.join(VDJBASE_SAMPLE)}") from None


# --- the manifest ------------------------------------------------------------------------------------------------

def fetch_manifest(data_root) -> dict:
    """What has been fetched under ``data_root``, from the records ``fetch_reads`` and ``fetch_genotype_files``
    wrote, written to ``<data root>/fetch_manifest.json``: the pins (``sra_tools_image``, ``igenotyper``), this
    module's ``code_sha256``, ``reads`` (run -> record), ``quality`` (run -> "full" | "lite"), ``genotypes``
    (donor -> file key -> record), ``cohort`` (``cohort_files``' records; None until fetched) and ``pending`` (the
    pilot runs and donors not fetched yet)."""
    root = Path(data_root)
    reads = {path.stem: _read_json(path) for path in sorted((root / RECORDS_DIR / "reads").glob("*.json"))}
    genotypes = {path.stem: _read_json(path) for path in sorted((root / RECORDS_DIR / "genotypes").glob("*.json"))}
    cohort = root / RECORDS_DIR / "cohort.json"
    manifest = {
        "kind": MANIFEST_KIND, "version": MANIFEST_VERSION, "data_root": str(root),
        "code_sha256": _CODE_SHA256, "sra_tools_image": SRA_TOOLS_IMAGE,
        "igenotyper": {"repository": IGENOTYPER_REPOSITORY, "commit": IGENOTYPER_COMMIT},
        "quality": {run: record["quality"] for run, record in reads.items()},
        "reads": reads, "genotypes": genotypes, "cohort": _read_json(cohort) if cohort.is_file() else None,
        "pending": {"reads": [run for run in PILOT_DONORS.values() if run not in reads],
                    "genotypes": [donor for donor in PILOT_DONORS if donor not in genotypes]},
    }
    _write_json(root / MANIFEST, manifest)
    return manifest


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, value) -> None:
    _write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def _write_text(path: Path, text: str) -> None:
    """Write ``text`` to ``path`` through a temporary file beside it, so a reader never sees half of it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)
