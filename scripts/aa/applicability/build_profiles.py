"""Stage-2 Step 6 — build the REAL production profile bank from pinned germline references.

Safeguards (owner-specified):
  * Pin every source: release, URL, raw-file SHA-256, retrieval time; reference the license-evidence hash.
  * Raw IMGT files are NOT shipped. Derived HMM profiles are redistributed under CC BY-4.0 with attribution
    + modification notice (see profiles.IMGT_CLEARANCE). OGRDB/Pfam profiles are CC0.
  * Functional, COMPLETE alleles only; translate from the annotated coding frame.
  * Reject internal stops, partials, invalid/ambiguous residues, and unresolved/uncodon-aligned frames.
  * Deduplicate identical PROTEIN sequences before MSA construction (allele multiplicity must not distort
    profile weighting).
  * Alignment method is pinned (source IMGT unique numbering, WithGaps reference); MSA and HMM are hashed.
  * Camelid profiles are labelled camelid_igh_v/j (non-human/camelid evidence), never "VHH".
  * V and J are paired only within the same species/locus group.
  * Per profile: input / accepted / rejected-by-reason / protein-distinct / MSA / HMM counts + hashes.
  * Raw references, MSAs, and HMM binaries stay PRIVATE; only the manifest + checksums are committed.

Run:  PYTHONPATH=src .venv/bin/python scripts/aa/applicability/build_profiles.py
"""
from __future__ import annotations

import datetime
import hashlib
import json
import re
import urllib.request
from pathlib import Path

from alignair.aa.guard.profiles import IMGT_CLEARANCE, _serialize_deterministic, build_profile

RAW = Path(".private/aa-applicability/germline_raw")
PROF = Path(".private/aa-applicability/profiles")
ART = Path("docs/superpowers/artifacts/2026-07-26-aa-applicability")
UA = {"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 Chrome/120 Safari/537.36"}
V_MIN_AA, J_MIN_AA = 85, 9        # completeness floors (ungapped) for V-REGION / J-REGION

# ---- genetic code -------------------------------------------------------------------------------------
_B = "TCAG"
_AA = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
CODON = {a + b + c: _AA[i] for i, (a, b, c) in enumerate((x, y, z) for x in _B for y in _B for z in _B)}


def _now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


def _sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def _download_pinned(url, dest):
    """Download once; return {url, retrieved_utc, sha256, bytes, path}."""
    if not dest.exists():
        req = urllib.request.Request(url, headers=UA)
        data = urllib.request.urlopen(req, timeout=180).read()
        dest.write_bytes(data)
    b = dest.read_bytes()
    return {"url": url, "retrieved_utc": _now(), "sha256": _sha_bytes(b), "bytes": len(b), "private_path": str(dest)}


def _parse_fasta(text):
    recs, h, s = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:].strip(), []
        elif line.strip():
            s.append(line.strip())
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def translate_gapped(nt, frame=1):
    """Translate an IMGT-gapped coding nt sequence codon-by-codon in the annotated frame. '...' -> '-'.
    Returns (gapped_aa, reject_reason|None)."""
    nt = nt[frame - 1:]
    aa = []
    for i in range(0, len(nt) - (len(nt) % 3), 3):
        cod = nt[i:i + 3]
        if cod == "...":
            aa.append("-")
            continue
        if "." in cod:
            return None, "gap_not_codon_aligned"
        cod = cod.upper()
        if any(x not in "ACGT" for x in cod):
            return None, "invalid_or_ambiguous_nt"
        r = CODON[cod]
        if r == "*" and i < len(nt) - 3:
            return None, "internal_stop"
        if r != "*":
            aa.append(r)
    return "".join(aa), None


_J_MOTIF = re.compile(r"[WF]G.G")     # conserved FR4/J motif: WGxG (heavy/TR) or FGxG (light/TRA)


def _translate_ungapped(nt, frame):
    nt = nt.replace(".", "").replace("-", "").upper()[frame - 1:]
    aa = []
    for i in range(0, len(nt) - (len(nt) % 3), 3):
        cod = nt[i:i + 3]
        if any(x not in "ACGT" for x in cod):
            return None
        aa.append(CODON[cod])
    return "".join(aa)


def _resolve_j_protein(nt):
    """Resolve a J allele's reading frame (OGRDB has no frame field) by the [WF]G.G motif with no internal
    stop; return the FR4 protein (motif kept, post-stop tail dropped) or None."""
    best = None
    for fr in (1, 2, 3):
        aa = _translate_ungapped(nt, fr)
        if aa is None:
            continue
        m = _J_MOTIF.search(aa)
        if m and "*" not in aa[:m.start() + 4]:
            trimmed = aa[:aa.find("*")] if "*" in aa else aa
            cand = (m.start(), trimmed)
            if best is None or cand[0] < best[0]:
                best = cand
    return best[1] if best else None


def process_j_frame_resolved(alleles):
    """OGRDB J path: resolve frame per allele, protein-dedup, then motif-anchor align into an MSA."""
    counts = {"input": len(alleles), "accepted": 0, "protein_distinct": 0,
              "rej_internal_stop": 0, "rej_partial": 0, "rej_invalid_nt": 0, "rej_gap_align": 0,
              "rej_unresolved_frame": 0}
    proteins = set()
    for _name, nt in alleles:
        aa = _resolve_j_protein(nt)
        if aa is None:
            counts["rej_unresolved_frame"] += 1; continue
        if len(aa) < J_MIN_AA:
            counts["rej_partial"] += 1; continue
        counts["accepted"] += 1
        proteins.add(aa)
    counts["protein_distinct"] = len(proteins)
    if not proteins:
        return None, counts, proteins
    entries = [(_J_MOTIF.search(aa).start(), aa) for aa in sorted(proteins)]
    max_pre = max(pre for pre, _ in entries)
    width = max(max_pre + (len(aa) - pre) for pre, aa in entries)
    lines = [f">j_{i}\n{('-' * (max_pre - pre) + aa).ljust(width, '-')[:width]}"
             for i, (pre, aa) in enumerate(entries)]
    return ("\n".join(lines) + "\n").encode(), counts, proteins


def process_alleles(alleles, region):
    """alleles: list[(name, gapped_nt, frame)] -> (msa_bytes, counts, protein_map). Protein-dedup before MSA."""
    floor = V_MIN_AA if region == "v" else J_MIN_AA
    counts = {"input": len(alleles), "accepted": 0, "protein_distinct": 0,
              "rej_internal_stop": 0, "rej_partial": 0, "rej_invalid_nt": 0, "rej_gap_align": 0}
    proteins = {}          # ungapped_aa -> gapped_aa (first occurrence)
    for name, nt, frame in alleles:
        aa, rej = translate_gapped(nt, frame)
        if rej == "internal_stop":
            counts["rej_internal_stop"] += 1; continue
        if rej == "invalid_or_ambiguous_nt":
            counts["rej_invalid_nt"] += 1; continue
        if rej == "gap_not_codon_aligned":
            counts["rej_gap_align"] += 1; continue
        ung = aa.replace("-", "")
        if len(ung) < floor:
            counts["rej_partial"] += 1; continue
        counts["accepted"] += 1
        proteins.setdefault(ung, aa)          # protein-dedup: one gapped exemplar per distinct protein
    counts["protein_distinct"] = len(proteins)
    if not proteins:
        return None, counts, proteins
    width = max(len(g) for g in proteins.values())
    lines = [f">{region}_{i}\n{g.ljust(width, '-')[:width]}" for i, g in enumerate(sorted(proteins.values()))]
    return ("\n".join(lines) + "\n").encode(), counts, proteins


# ---- source extractors --------------------------------------------------------------------------------
def imgt_alleles(recs, species, locus_prefix, region_label):
    out = []
    for h, seq in recs:
        f = h.split("|")
        if len(f) > 7 and species in f[2] and f[1].startswith(locus_prefix) and f[3] == "F" and f[4] == region_label:
            frame = int(f[7]) if f[7].strip().isdigit() else 1
            out.append((f[1], seq, frame))
    return out


def ogrdb_alleles(recs, gene_prefix):
    # OGRDB 'published' gapped sets are curated functional; headers are '>gene*allele', IMGT-numbered (frame 1)
    return [(h.split()[0], seq, 1) for h, seq in recs if h.split()[0].startswith(gene_prefix)]


# ---- profile plan: (group, region, source, label, extractor-args) -------------------------------------
def build():
    RAW.mkdir(parents=True, exist_ok=True)
    PROF.mkdir(parents=True, exist_ok=True)

    sources, manifest_profiles = {}, {}

    # IMGT GENE-DB (CC BY-4.0) — one pinned file, all TR + mouse IGH + camelid IGH
    imgt = _download_pinned(
        "https://www.imgt.org/download/GENE-DB/IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F%2BORF%2BinframeP",
        RAW / "IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP")
    imgt["license"] = "CC-BY-4.0"; imgt["attribution_required"] = True
    imgt["evidence"] = IMGT_CLEARANCE["evidence_archive"]
    sources["imgt_gene_db"] = imgt
    imgt_recs = _parse_fasta((RAW / "IMGTGENEDB-ReferenceSequences.fasta-nt-WithGaps-F+ORF+inframeP")
                             .read_bytes().decode("utf-8", "replace"))

    IMGT_PLAN = [
        ("tra", "v", "Homo sapiens", "TRAV"), ("tra", "j", "Homo sapiens", "TRAJ"),
        ("trb", "v", "Homo sapiens", "TRBV"), ("trb", "j", "Homo sapiens", "TRBJ"),
        ("trg", "v", "Homo sapiens", "TRGV"), ("trg", "j", "Homo sapiens", "TRGJ"),
        ("trd", "v", "Homo sapiens", "TRDV"), ("trd", "j", "Homo sapiens", "TRDJ"),
        ("murine_vh", "v", "Mus musculus", "IGHV"), ("murine_vh", "j", "Mus musculus", "IGHJ"),
        # camelid IGH germline (IMGT Vicugna pacos); profile identity is camelid_igh — non-human/camelid
        # evidence, NOT a biologically definitive "VHH" profile
        ("camelid_igh", "v", "Vicugna pacos", "IGHV"), ("camelid_igh", "j", "Vicugna pacos", "IGHJ"),
    ]
    for group, region, species, prefix in IMGT_PLAN:
        reglab = "V-REGION" if region == "v" else "J-REGION"
        alleles = imgt_alleles(imgt_recs, species, prefix, reglab)
        msa, counts, _ = process_alleles(alleles, region)
        _persist(manifest_profiles, group, region, msa, counts, "imgt_gene_db", "CC-BY-4.0",
                 f"IMGT {species} {prefix} {reglab} (F)", group, PROF,
                 "IMGT unique numbering (WithGaps reference); annotated frame per allele")

    # OGRDB (CC0) — human IGH/IGK/IGL, per-set gapped download
    OGRDB_SETS = {"human_igh": ("Homo%20sapiens", "IGH_VDJ", {"v": "IGHV", "j": "IGHJ"}),
                  "human_igk": ("Homo%20sapiens", "IGKappa_VJ", {"v": "IGKV", "j": "IGKJ"}),
                  "human_igl": ("Homo%20sapiens", "IGLambda_VJ", {"v": "IGLV", "j": "IGLJ"})}
    for group, (sp, setname, prefixes) in OGRDB_SETS.items():
        url = f"https://ogrdb.airr-community.org/download_germline_set/{sp}/{setname}/published/gapped"
        dest = RAW / f"ogrdb_{group}_gapped.fasta"
        try:
            src = _download_pinned(url, dest)
        except Exception as e:  # noqa: BLE001
            manifest_profiles.setdefault(group, {})["_error"] = f"download_failed:{type(e).__name__}"
            continue
        src["license"] = "CC0-1.0"; src["attribution_required"] = False
        sources[f"ogrdb_{group}"] = src
        recs = _parse_fasta(dest.read_bytes().decode("utf-8", "replace"))
        for region, prefix in prefixes.items():
            raw = [(h.split()[0], seq) for h, seq in recs if h.split()[0].startswith(prefix)]
            if region == "j":
                # OGRDB download carries no frame field and J genes are not frame-1 -> resolve per allele
                msa, counts, _ = process_j_frame_resolved(raw)
                am = "reading-frame-resolved ([WF]G.G motif, no internal stop) + motif-anchor alignment"
            else:
                msa, counts, _ = process_alleles([(n, s, 1) for n, s in raw], "v")
                am = "IMGT unique numbering (OGRDB published/gapped, frame 1)"
            _persist(manifest_profiles, group, region, msa, counts, f"ogrdb_{group}", "CC0-1.0",
                     f"OGRDB {group} {prefix} published/gapped", group, PROF, am)

    # Pfam PF07686 V-set full-domain HMM (CC0) — used directly (already an HMM), decompressed + persisted
    try:
        import gzip
        pf = _download_pinned("https://www.ebi.ac.uk/interpro/wwwapi/entry/pfam/PF07686/?annotation=hmm",
                              RAW / "PF07686.hmm.gz")
        pf["license"] = "CC0"; pf["attribution_required"] = False
        sources["pfam_pf07686"] = pf
        hmm_bytes = gzip.decompress((RAW / "PF07686.hmm.gz").read_bytes())
        (PROF / "igsf_vset_domain.hmm").write_bytes(hmm_bytes)
        acc = next((ln.split()[1] for ln in hmm_bytes.decode("latin1").splitlines() if ln.startswith("ACC")), "PF07686")
        manifest_profiles.setdefault("igsf_vset", {})["domain"] = {
            "source": "pfam_pf07686", "license": "CC0", "label": "pfam_v_set_PF07686", "region_label": "domain",
            "pfam_accession": acc, "raw_gz_sha256": pf["sha256"], "hmm_sha256": _sha_bytes(hmm_bytes),
            "note": "Pfam-A HMM used directly (CC0, no attribution); full-domain V-set profile"}
    except Exception as e:  # noqa: BLE001
        manifest_profiles.setdefault("igsf_vset", {})["domain"] = {"_error": f"download_failed:{type(e).__name__}"}

    manifest = {
        "artifact": "profile_bank_manifest", "built_utc": _now(),
        "alignment_method": "IMGT unique numbering (source WithGaps reference); protein-dedup before build; "
                            "pyhmmer Builder seed=42; DATE/COM neutralized for determinism",
        "hmmer_runtime": "pyhmmer 0.12.1 / HMMER 3.4",
        "ship_policy": "raw germline files are NOT shipped; only derived HMM profiles are redistributed "
                       "(CC BY-4.0 with attribution+modification-notice for IMGT-derived; CC0 for OGRDB/Pfam)",
        "sources": sources, "profiles": manifest_profiles,
    }
    bank_parts = [f"{g}:{r}:{v.get('hmm_sha256', v.get('raw_sha256', '-'))}"
                  for g in sorted(manifest_profiles) for r, v in sorted(manifest_profiles[g].items())
                  if isinstance(v, dict)]
    manifest["bank_checksum"] = _sha_bytes("\n".join(bank_parts).encode())
    (ART / "profile_bank_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(json.dumps({"bank_checksum": manifest["bank_checksum"],
                      "profiles": {g: {r: (m[r].get("accepted") if isinstance(m[r], dict) else None)
                                       for r in m if isinstance(m[r], dict)} for g, m in manifest_profiles.items()}},
                     indent=2, sort_keys=True))


def _persist(manifest_profiles, group, region, msa, counts, source, license_, region_label, group_label,
             prof_dir, align_method):
    """Build the HMM from a precomputed MSA, persist HMM+MSA privately, record manifest counts+hashes."""
    entry = {"source": source, "license": license_, "label": f"{group_label}_{region}",
             "region_label": region_label, "alignment_method": align_method, "counts": counts}
    if msa is None:
        entry["_error"] = "no_accepted_alleles"
        manifest_profiles.setdefault(group, {})[region] = entry
        return
    hmm, hmm_sha = build_profile(group, region, msa)
    # persist the DETERMINISTIC (date-neutralized) bytes so the private file's sha == manifest hmm_sha256
    (prof_dir / f"{group}_{region}.hmm").write_bytes(_serialize_deterministic(hmm))
    (prof_dir / f"{group}_{region}.msa.fasta").write_bytes(msa)   # private MSA (reproducibility)
    entry["msa_sha256"] = _sha_bytes(msa)
    entry["hmm_sha256"] = hmm_sha
    entry["hmm_M"] = hmm.M
    manifest_profiles.setdefault(group, {})[region] = entry


if __name__ == "__main__":
    build()
