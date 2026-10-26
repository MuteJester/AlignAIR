import gzip
import hashlib
import json

import pytest

from alignair.aa.guard.profiles import build_profile, load_bank
from alignair.aa.guard.profiles import alignment, builder, cli
from alignair.aa.guard.profiles.bank import _serialize_deterministic
from alignair.aa.translation import CODON_TABLE
from alignair.errors import ArtifactIntegrityError, DataContractError

pytestmark = pytest.mark.component


def _nt(protein):
    codons = {aa: codon for codon, aa in CODON_TABLE.items()}
    return "".join(codons[aa] for aa in protein)


@pytest.mark.parametrize("sequence,frame,protein,reason", [
    ("ATG...TGG", 1, "M-W", None), ("aatg...tgg", 2, "M-W", None),
    ("AAATGTGG", 3, "MW", None), ("ATGTAA", 1, "M", None),
    ("ATGTAATGG", 1, None, "internal_stop"), ("ATGNNN", 1, None, "invalid_or_ambiguous_nt"),
    ("ATGA..", 1, None, "gap_not_codon_aligned"), ("ATGA", 1, "M", None),
    ("ATG", 0, None, "invalid_frame"), ("ATG", None, None, "invalid_frame"),
])
def test_gapped_translation_honors_frame_and_preserves_terminal_partial_codon_policy(sequence, frame, protein, reason):
    assert alignment.translate_gapped(sequence, frame) == (protein, reason)


def test_protein_dedup_precedes_alignment_and_rejections_are_counted():
    sequence = _nt("A" * 85)
    msa, counts, proteins = alignment.process_alleles([
        ("first", sequence, 1), ("same-protein", "..." + sequence, 1),
        ("stop", sequence[:9] + "TAA" + sequence[12:], 1),
        ("short", "ATG", 1), ("bad-frame", sequence, None),
    ], "v")
    assert counts["input"] == 5 and counts["accepted"] == 2 and counts["protein_distinct"] == 1
    assert counts["rej_internal_stop"] == counts["rej_partial"] == counts["rej_invalid_frame"] == 1
    assert proteins == {"A" * 85: "A" * 85}
    assert msa == (">v_0\n" + "A" * 85 + "\n").encode()


def test_j_frame_resolution_aligns_conserved_motif_after_protein_dedup():
    protein = "WGQGTLVTVSS"
    msa, counts, proteins = alignment.process_j_frame_resolved([
        ("frame2", "A" + _nt(protein)), ("frame3", "AA" + _nt(protein)),
        ("long-prefix", _nt("AA" + protein)), ("unresolved", "NNNNNN"),
    ])
    assert counts["accepted"] == 3 and counts["protein_distinct"] == 2
    assert counts["rej_unresolved_frame"] == 1
    assert proteins == {protein, "AA" + protein}
    rows = msa.decode().splitlines()[1::2]
    assert len({len(row) for row in rows}) == 1
    assert {row.index("WGQG") for row in rows} == {2}


@pytest.fixture
def sources(tmp_path, monkeypatch):
    import urllib.request

    def refuse(*args, **kwargs):
        pytest.fail("Pinned local profile builds must not make network requests.")

    monkeypatch.setattr(urllib.request, "urlopen", refuse)
    v, j = _nt("A" * 85), _nt("WGQGTLVTVSS")
    imgt = []
    for _, species, vp, jp in builder.IMGT_PLAN:
        for prefix, region, sequence in ((vp, "V-REGION", v), (jp, "J-REGION", j)):
            header = f"accession|{prefix}1*01|{species}|F|{region}|1..300|300 nt|1"
            imgt.append(f">{header}\n{sequence}\n")
    raw = {"imgt_gene_db": "".join(imgt).encode()}
    for group, (vp, jp) in builder.OGRDB_PLAN.items():
        raw["ogrdb_" + group] = f">{vp}1*01\n{v}\n>{jp}1*01\nA{j}\n".encode()
    hmm, _ = build_profile("igsf_vset", "domain", b">v\nACDEFGHIKLMNPQRSTVWY\n")
    raw["pfam_pf07686"] = gzip.compress(_serialize_deterministic(hmm), mtime=0)
    specifications = {}
    for key, data in raw.items():
        path = tmp_path / key
        path.write_bytes(data)
        specifications[key] = dict(path=str(path), sha256=hashlib.sha256(data).hexdigest(),
                                   url=f"https://example.org/{key}", version="fixture-release-1")
    return specifications


def test_complete_bank_reloads_with_matching_file_hashes_and_reproducible_profiles(sources, tmp_path):
    first, second = tmp_path / "bank1", tmp_path / "bank2"
    manifest = builder.build_profile_bank(sources, output_dir=first)
    repeated = builder.build_profile_bank(sources, output_dir=second)
    assert manifest["bank_checksum"] == repeated["bank_checksum"] == load_bank(first).checksum()
    assert manifest["profiles"] == repeated["profiles"]
    assert len(list(first.glob("*.hmm"))) == 19
    assert len(list(first.glob("*.msa.fasta"))) == 18
    assert json.loads((first / "profile_bank_manifest.json").read_text()) == manifest
    for group, regions in manifest["profiles"].items():
        for region, profile in regions.items():
            assert hashlib.sha256((first / f"{group}_{region}.hmm").read_bytes()).hexdigest() == profile["hmm_sha256"]
    assert not any(path.name in builder.SOURCE_KEYS for path in first.iterdir())


@pytest.mark.parametrize("failure", ["checksum", "missing-profile", "invalid-pfam"])
def test_bad_or_incomplete_inputs_cannot_publish_a_partial_bank(sources, tmp_path, failure):
    if failure == "checksum":
        sources["imgt_gene_db"]["sha256"] = "0" * 64
    else:
        key = "ogrdb_human_igh" if failure == "missing-profile" else "pfam_pf07686"
        data = b">IGHV1*01\nATG\n" if failure == "missing-profile" else b"not a gzip file"
        from pathlib import Path
        Path(sources[key]["path"]).write_bytes(data)
        sources[key]["sha256"] = hashlib.sha256(data).hexdigest()
    output = tmp_path / "invalid-bank"
    with pytest.raises((ArtifactIntegrityError, DataContractError)):
        builder.build_profile_bank(sources, output_dir=output)
    assert not output.exists()


def test_existing_output_directory_is_refused_before_reading_sources(tmp_path):
    output = tmp_path / "bank"
    output.mkdir()
    marker = output / "existing.hmm"
    marker.write_bytes(b"preserve")
    with pytest.raises(ArtifactIntegrityError):
        builder.build_profile_bank({}, output_dir=output)
    assert marker.read_bytes() == b"preserve"


def test_command_resolves_relative_source_paths_and_builds_a_verified_bank(sources, tmp_path, capsys):
    from pathlib import Path
    for spec in sources.values():
        spec["path"] = Path(spec["path"]).name
    config = tmp_path / "sources.json"
    config.write_text(json.dumps(sources))
    output = tmp_path / "command-bank"
    assert cli.main(["--sources", str(config), "--out", str(output)]) == 0
    assert json.loads(capsys.readouterr().out)["bank_checksum"] == load_bank(output).checksum()
