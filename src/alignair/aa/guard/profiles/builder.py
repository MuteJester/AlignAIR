"""Build complete profile banks from explicitly pinned local germline sources."""
from __future__ import annotations

import datetime
import gzip
import hashlib
import io
import json
from importlib.metadata import version
from pathlib import Path

from alignair.errors import ArtifactIntegrityError, ConfigurationError, DataContractError
from .alignment import _parse_fasta, imgt_alleles, ogrdb_alleles, process_alleles, process_j_frame_resolved
from .bank import ProfileBank, _serialize_deterministic, build_profile
from .sources import IMGT_CLEARANCE, resolve_production_bank

SOURCE_KEYS = ("imgt_gene_db", "ogrdb_human_igh", "ogrdb_human_igk", "ogrdb_human_igl", "pfam_pf07686")
IMGT_PLAN = (
    ("tra", "Homo sapiens", "TRAV", "TRAJ"), ("trb", "Homo sapiens", "TRBV", "TRBJ"),
    ("trg", "Homo sapiens", "TRGV", "TRGJ"), ("trd", "Homo sapiens", "TRDV", "TRDJ"),
    ("murine_vh", "Mus musculus", "IGHV", "IGHJ"), ("camelid_igh", "Vicugna pacos", "IGHV", "IGHJ"),
)
OGRDB_PLAN = {"human_igh": ("IGHV", "IGHJ"), "human_igk": ("IGKV", "IGKJ"), "human_igl": ("IGLV", "IGLJ")}
MAX_SOURCE_BYTES = 128 << 20
MAX_PFAM_BYTES = 16 << 20


def _sha(raw):
    return hashlib.sha256(raw).hexdigest()


def _read_sources(sources):
    if not isinstance(sources, dict) or set(sources) != set(SOURCE_KEYS):
        raise ConfigurationError("Profile build requires the five declared germline/HMM sources.",
                                 context={"required_sources": SOURCE_KEYS})
    snapshots, provenance = {}, {}
    for key in SOURCE_KEYS:
        spec = sources[key]
        if not isinstance(spec, dict) or any(not isinstance(spec.get(field), str) or not spec[field]
                                              for field in ("path", "url", "version", "sha256")):
            raise DataContractError("Profile source requires path, URL, release/version, and expected SHA-256.",
                                    context={"source": key})
        expected = spec["sha256"]
        if len(expected) != 64 or any(char not in "0123456789abcdef" for char in expected):
            raise DataContractError("Expected source checksum must be a SHA-256 hex string.", context={"source": key})
        path = Path(spec["path"])
        with path.open("rb") as stream:
            raw = stream.read(MAX_SOURCE_BYTES + 1)
        if len(raw) > MAX_SOURCE_BYTES:
            raise DataContractError("Germline source exceeds its byte limit.", context={"path": path})
        if _sha(raw) != expected:
            raise ArtifactIntegrityError("Germline source differs from its expected checksum.", context={"source": key, "path": path})
        snapshots[key] = raw
        provenance[key] = {"url": spec["url"], "version": spec["version"], "sha256": expected,
                           "bytes": len(raw), "private_path": str(path)}
    return snapshots, provenance


def _fasta(raw, source):
    try:
        return _parse_fasta(raw.decode("utf-8"))
    except UnicodeDecodeError as exc:
        raise DataContractError("Germline FASTA is not UTF-8 text.", context={"source": source}) from exc


def build_profile_bank(sources, *, output_dir):
    """Validate local inputs, build every required profile, and write a new bank.

    Each source supplies ``path``, ``url``, ``version``, and an expected ``sha256``.
    No downloads occur. Existing output directories are refused. Raw sources are
    never copied into the output; profile HMMs, alignments, and a manifest are written.
    The manifest is written last, after all required profiles were built successfully.
    """
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise ArtifactIntegrityError("Profile bank output already exists.", context={"path": output_dir})
    resolve_production_bank()
    snapshots, provenance = _read_sources(sources)
    profiles, entries, files = {}, {}, {}

    def add(group, region, msa, counts, source, license_, region_label, alignment_method):
        if msa is None:
            raise DataContractError("No accepted germline alleles for a required profile.",
                                    context={"group": group, "region": region, "counts": counts})
        hmm, checksum = build_profile(group, region, msa)
        raw = _serialize_deterministic(hmm)
        profiles.setdefault(group, {})[region] = (hmm, checksum)
        files[f"{group}_{region}.hmm"] = raw
        files[f"{group}_{region}.msa.fasta"] = msa
        entries.setdefault(group, {})[region] = {
            "source": source, "license": license_, "label": f"{group}_{region}", "region_label": region_label,
            "alignment_method": alignment_method, "counts": counts,
            "msa_sha256": _sha(msa), "hmm_sha256": checksum, "hmm_M": hmm.M,
        }

    provenance["imgt_gene_db"].update(license="CC-BY-4.0", attribution_required=True,
                                      evidence=IMGT_CLEARANCE["evidence_archive"])
    imgt = _fasta(snapshots["imgt_gene_db"], "imgt_gene_db")
    for group, species, v_prefix, j_prefix in IMGT_PLAN:
        for region, prefix in (("v", v_prefix), ("j", j_prefix)):
            label = "V-REGION" if region == "v" else "J-REGION"
            msa, counts, _ = process_alleles(imgt_alleles(imgt, species, prefix, label), region)
            add(group, region, msa, counts, "imgt_gene_db", "CC-BY-4.0",
                f"IMGT {species} {prefix} {label} (F)", "IMGT unique numbering (WithGaps reference); annotated frame per allele")
    for group, prefixes in OGRDB_PLAN.items():
        key = "ogrdb_" + group
        provenance[key].update(license="CC0-1.0", attribution_required=False)
        records = _fasta(snapshots[key], key)
        for region, prefix in zip(("v", "j"), prefixes):
            alleles = ogrdb_alleles(records, prefix)
            if region == "j":
                msa, counts, _ = process_j_frame_resolved([(name, sequence) for name, sequence, _ in alleles])
                method = "reading-frame-resolved ([WF]G.G motif, no internal stop) + motif-anchor alignment"
            else:
                msa, counts, _ = process_alleles(alleles, region)
                method = "IMGT unique numbering (OGRDB published/gapped, frame 1)"
            add(group, region, msa, counts, key, "CC0-1.0", f"OGRDB {group} {prefix} published/gapped", method)

    from pyhmmer import plan7

    try:
        with gzip.GzipFile(fileobj=io.BytesIO(snapshots["pfam_pf07686"])) as stream:
            pfam = stream.read(MAX_PFAM_BYTES + 1)
    except (OSError, EOFError) as exc:
        raise DataContractError("Invalid compressed Pfam profile.") from exc
    if len(pfam) > MAX_PFAM_BYTES:
        raise DataContractError("Decompressed Pfam profile exceeds its byte limit.")
    try:
        with plan7.HMMFile(io.BytesIO(pfam)) as stream:
            hmm = stream.read()
            if hmm is None or stream.read() is not None or not hmm.alphabet.is_amino():
                raise DataContractError("Pfam source must contain exactly one amino-acid HMM.")
    except (ValueError, EOFError) as exc:
        raise DataContractError("Invalid Pfam HMM data.") from exc
    profiles["igsf_vset"] = {"domain": (hmm, _sha(pfam))}
    files["igsf_vset_domain.hmm"] = pfam
    provenance["pfam_pf07686"].update(license="CC0", attribution_required=False)
    entries["igsf_vset"] = {"domain": {
        "source": "pfam_pf07686", "license": "CC0", "label": "pfam_v_set_PF07686", "region_label": "domain",
        "pfam_accession": (hmm.accession or b"PF07686").decode(),
        "raw_gz_sha256": provenance["pfam_pf07686"]["sha256"], "hmm_sha256": _sha(pfam),
        "note": "Pfam-A HMM used directly (CC0, no attribution); full-domain V-set profile",
    }}
    bank = ProfileBank(profiles)
    manifest = {
        "artifact": "profile_bank_manifest", "built_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "alignment_method": "IMGT unique numbering (source WithGaps reference); protein-dedup before build; "
                            "pyhmmer Builder seed=42; DATE/COM neutralized for determinism",
        "hmmer_runtime": "pyhmmer " + version("pyhmmer"),
        "ship_policy": "raw germline files are NOT shipped; only derived HMM profiles are redistributed "
                       "(CC BY-4.0 with attribution+modification-notice for IMGT-derived; CC0 for OGRDB/Pfam)",
        "sources": provenance, "profiles": entries, "bank_checksum": bank.checksum(),
    }
    # Exclusive directory creation prevents mixing a rebuilt bank with stale HMMs.
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    try:
        output_dir.mkdir()
    except FileExistsError as exc:
        raise ArtifactIntegrityError("Profile bank output already exists.", context={"path": output_dir}) from exc
    for name, raw in files.items():
        (output_dir / name).write_bytes(raw)
    (output_dir / "profile_bank_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest
