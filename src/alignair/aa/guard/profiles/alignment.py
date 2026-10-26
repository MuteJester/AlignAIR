"""Germline allele selection, translation, protein deduplication, and alignment."""
from __future__ import annotations

import re

from alignair.aa.translation import CODON_TABLE
from alignair.errors import ConfigurationError, DataContractError

V_MIN_AA, J_MIN_AA = 85, 9

def _parse_fasta(text):
    recs, h, s = [], None, []
    for line in text.splitlines():
        if line.startswith(">"):
            if h is not None:
                recs.append((h, "".join(s)))
            h, s = line[1:].strip(), []
            if not h:
                raise DataContractError("Germline FASTA header is empty.")
        elif line.strip():
            if h is None:
                raise DataContractError("Germline sequence occurs before its FASTA header.")
            s.append(line.strip())
    if h is not None:
        recs.append((h, "".join(s)))
    return recs


def translate_gapped(nt, frame=1):
    """Translate an IMGT-gapped coding nt sequence codon-by-codon in the annotated frame. '...' -> '-'.
    Returns (gapped_aa, reject_reason|None)."""
    if isinstance(frame, bool) or not isinstance(frame, int) or frame not in (1, 2, 3):
        return None, "invalid_frame"
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
        r = CODON_TABLE[cod]
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
        aa.append(CODON_TABLE[cod])
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
    if region not in ("v", "j"):
        raise ConfigurationError("Germline allele region must be v or j.")
    floor = V_MIN_AA if region == "v" else J_MIN_AA
    counts = {"input": len(alleles), "accepted": 0, "protein_distinct": 0,
              "rej_internal_stop": 0, "rej_partial": 0, "rej_invalid_nt": 0, "rej_gap_align": 0,
              "rej_invalid_frame": 0}
    proteins = {}          # ungapped_aa -> gapped_aa (first occurrence)
    for name, nt, frame in alleles:
        aa, rej = translate_gapped(nt, frame)
        if rej == "invalid_frame":
            counts["rej_" + rej] += 1
            continue
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
            frame = int(f[7]) if f[7].strip().isdigit() else None
            out.append((f[1], seq, frame))
    return out


def ogrdb_alleles(recs, gene_prefix):
    # OGRDB 'published' gapped sets are curated functional; headers are '>gene*allele', IMGT-numbered (frame 1)
    return [(h.split()[0], seq, 1) for h, seq in recs if h.split()[0].startswith(gene_prefix)]
