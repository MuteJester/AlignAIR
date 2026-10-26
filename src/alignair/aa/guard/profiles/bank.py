"""Deterministic HMM construction, loading, and profile-bank identities."""
from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass

# Deterministic profile builder (pyhmmer) -------------------------------------------------
# Receptor groups carry independently versioned V and J profiles; a complete domain needs BOTH (V-only
# supports locus evidence but never a complete domain). IgSF uses a validated full-domain profile. Builds
# are deterministic: fixed builder seed + the varying HMMER DATE/COM header lines neutralized before
# serialization, so the same MSA yields the same sha256 across runs (verified Phase 0).
_BUILDER_SEED = 42
_AMINO = None


def _amino():
    """Cached singleton amino alphabet. MUST be shared across build AND search: an HMM built with one
    Alphabet instance fails p7_ProfileConfig (eslEINVAL) when searched with a Pipeline made from a
    different instance."""
    global _AMINO
    if _AMINO is None:
        from pyhmmer import easel
        _AMINO = easel.Alphabet.amino()
    return _AMINO


def _digitize_msa(aligned_fasta_bytes: bytes, name: bytes):
    from pyhmmer import easel
    records, cur, seq = [], None, []
    for line in aligned_fasta_bytes.decode("utf-8").splitlines():
        if line.startswith(">"):
            if cur is not None:
                records.append((cur, "".join(seq)))
            cur, seq = line[1:].strip(), []
        elif line.strip():
            seq.append(line.strip())
    if cur is not None:
        records.append((cur, "".join(seq)))
    abc = _amino()
    msa = easel.TextMSA(name=name,
                        sequences=[easel.TextSequence(name=n.encode(), sequence=s) for n, s in records])
    return msa.digitize(abc)


def _serialize_deterministic(hmm) -> bytes:
    # Serialize a COPY: hmm.write() leaves the object un-searchable (p7_ProfileConfig eslEINVAL), so the
    # original must be returned untouched for pyhmmer search. The copy is date/command-neutralized so the
    # same MSA yields the same bytes (hence sha256) across runs.
    c = hmm.copy()
    c.creation_time = None        # drop the run-varying DATE line
    c.command_line = None         # stable COM placeholder (no varying content)
    buf = io.BytesIO()
    c.write(buf)
    return buf.getvalue()


def build_profile(group: str, region: str, aligned_fasta_bytes: bytes):
    """Build one deterministic profile HMM from an aligned FASTA MSA; return (HMM, sha256_hex of bytes)."""
    from pyhmmer import plan7
    abc = _amino()
    dmsa = _digitize_msa(aligned_fasta_bytes, name=f"{group}:{region}".encode())
    builder = plan7.Builder(abc, seed=_BUILDER_SEED)
    hmm, _, _ = builder.build_msa(dmsa, plan7.Background(abc))
    data = _serialize_deterministic(hmm)
    return hmm, hashlib.sha256(data).hexdigest()


@dataclass
class ProfileBank:
    """In-memory profile bank: group -> region -> (HMM, sha256_hex). Persisted elsewhere via pyhmmer
    HMMFile write (non-pickle); the checksum is over the per-profile hashes only."""
    profiles: dict

    def supports_complete_domain(self, group: str) -> bool:
        regs = set(self.profiles.get(group, {}))
        if group == "igsf_vset":
            return "domain" in regs
        return {"v", "j"} <= regs         # a complete receptor domain needs BOTH V and J profiles

    def checksum(self) -> str:
        parts = [f"{g}:{r}:{self.profiles[g][r][1]}"
                 for g in sorted(self.profiles) for r in sorted(self.profiles[g])]
        return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def build_bank_fixture(regions: dict) -> ProfileBank:
    """Build a small in-memory bank from {group: {region: aligned_fasta_bytes}} (burned-dev fixtures/tests)."""
    profiles = {g: {r: build_profile(g, r, msa) for r, msa in regs.items()} for g, regs in regions.items()}
    return ProfileBank(profiles=profiles)


def load_bank(profiles_dir) -> ProfileBank:
    """Load a persisted bank of ``<group>_<region>.hmm`` files (pyhmmer HMMFile, non-pickle) into a
    ProfileBank; each profile's sha is the sha256 of its file bytes. Loaded HMMs search under the shared
    amino alphabet."""
    import hashlib
    import pathlib

    from pyhmmer import plan7
    _amino()                              # ensure the shared alphabet is initialized
    profiles: dict = {}
    for p in sorted(pathlib.Path(profiles_dir).glob("*.hmm")):
        group, region = p.stem.rsplit("_", 1)          # e.g. human_igh_v -> (human_igh, v); igsf_vset_domain
        with plan7.HMMFile(str(p)) as hf:
            hmm = hf.read()
        profiles.setdefault(group, {})[region] = (hmm, hashlib.sha256(p.read_bytes()).hexdigest())
    return ProfileBank(profiles=profiles)
