"""The ``real`` workflow (``workflow.real``): refusals before any work, the manifest and ``--force``, resumable steps,
a donor stopped by its preprocessing, cluster-aware comparisons and the pooled summary.

The offline tests replace the steps that need Docker, the data root or the network with fakes
(``real_steps.check_images``, ``fetch_inputs``, ``capture_genotype``, ``preprocess_run``, ``load_reference``,
``load_iuis``, ``sensitivity_genes``, ``fetch_manifest``) and register fake tools under the three tool names. The
reads, the genotype and the reference are synthetic, and the truth is computed for real from them. The smoke test
runs W-97 end to end on 500 reads when Docker, the data root and the checkpoint are all given; it is slow.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import os
import random
import time
from collections import Counter
from pathlib import Path

import pytest

from alignair_assay.cli import main
from alignair_assay.core.cases import read_fasta
from alignair_assay.core.container import ContainerError
from alignair_assay.core.layout import RunLayout
from alignair_assay.core.reference import Reference
from alignair_assay.real.genotype import IuisMap, RealAllele, RealGenotype
from alignair_assay.real.preprocess import TSV_COLUMNS, PreprocessError
from alignair_assay.real.truth import cys104_end
from alignair_assay.runners import RUNNERS
from alignair_assay.runners.base import RunResult
from alignair_assay.workflow import real_score, real_steps
from alignair_assay.workflow.real import RealConfig, run_real

DATA = os.environ.get("ALIGNAIR_ASSAY_DATA")
DOCKER = os.environ.get("ALIGNAIR_ASSAY_DOCKER") == "1"
MODEL = os.environ.get("ALIGNAIR_ASSAY_MODEL")
IGBLAST = Path("/home/thomas/Desktop/AlignAIR/.private/tools/ncbi-igblast-1.22.0")

# IMGT gap columns (0-based, half-open) of the synthetic V alleles: codon 104 (columns 309-311) is TGT.
GAPS = ((27, 30), (72, 84), (165, 171), (195, 198))
TAIL = "TGGGGCCAGGGAACCCTGGTCACCGTCTCCTCAG"  # W118 G119 Q120 G121 ... (an IGHJ4 end)


def _random(n: int, seed: int, alphabet: str = "ACGT") -> str:
    return "".join(random.Random(seed).choices(alphabet, k=n))


def _gapped(seed: int, *mutated: int) -> str:
    bases = list(_random(320, seed))
    bases[309:312] = "TGT"
    for column in mutated:
        bases[column] = "A" if bases[column] != "A" else "C"
    for start, end in GAPS:
        bases[start:end] = "." * (end - start)
    return "".join(bases)


# Shared V allele -> (IMGT-gapped sequence, IUIS labels with its sequence).
SHARED_V = {"IGHVF1-G1*01": (_gapped(1), ("IGHV1-2*02",)), "IGHVF1-G1*02": (_gapped(1, 150), ("IGHV1-2*04",)),
            "IGHVF7-G1*01": (_gapped(7), ("IGHV3-11*01",)), "IGHVF5-G1*01": (_gapped(5), ("IGHV3-23*01",))}
REFERENCE = Reference(
    alleles={"v": {name: gapped.replace(".", "") for name, (gapped, _) in SHARED_V.items()},
             "d": {"IGHD3-10*01": _random(31, 20)}, "j": {"IGHJ4*02": _random(48, 30), "IGHJ6*02": _random(63, 31)}},
    anchors={}, gapped_v={name: gapped for name, (gapped, _) in SHARED_V.items()})


def _iuis() -> IuisMap:
    labels = {**{name: labels for name, (_, labels) in SHARED_V.items()},
              **{name: (name,) for table in ("d", "j") for name in REFERENCE.alleles[table]}}
    sequences = REFERENCE.sequences()
    by_sequence = {sequences[name]: tuple(sorted(names)) for name, names in labels.items()}
    return IuisMap(labels_by_shared={name: by_sequence[sequence] for name, sequence in sequences.items()},
                   shared_by_sequence={sequence: (name,) for name, sequence in sequences.items()},
                   labels_by_sequence=by_sequence, reference_digest=REFERENCE.digest(),
                   source={"sha256": "ogrdb", "release_version": 8})


IUIS = _iuis()
V1, V311 = REFERENCE.seq("IGHVF1-G1*01"), REFERENCE.seq("IGHVF7-G1*01")
CYS_END = cys104_end(SHARED_V["IGHVF1-G1*01"][0])


def _genotype(donor: str, captured=frozenset({"IGHV1-2", "IGHV3-11"})) -> RealGenotype:
    """The donor carries IGHV1-2*02 on both haplotypes; IGHV3-11 is captured and not carried."""
    alleles = tuple(RealAllele(allele_id="IGHV1-2*02", gene="IGHV1-2", segment="v", sequence=V1,
                               gapped=SHARED_V["IGHVF1-G1*01"][0], haplotype=haplotype, functional=True,
                               shared=("IGHVF1-G1*01",), iuis=("IGHV1-2*02",)) for haplotype in (1, 2))
    return RealGenotype(donor=donor, alleles=alleles, captured=frozenset(captured), resolved=frozenset(captured),
                        source={"kind": "test", "sha256": "csv", "reference_digest": REFERENCE.digest()})


def _reads(donor: str) -> list[dict]:
    """Twelve IGHV1-2 clones of two reads (a known answer and a mutated member; IgM in the first six, so six naive
    clonotypes) and three IGHV3-11 clones of two reads, which only the shared reference's anchors place."""
    rows = []
    for k in range(12):
        cdr3 = _random(18, 100 + k, "ACT")
        exact = V1[:CYS_END] + cdr3 + TAIL
        mutated = exact[:10] + ("A" if exact[10] != "A" else "C") + exact[11:]
        isotype = "IGHM" if k < 6 else "IGHG"
        rows += [{"sequence_id": f"v1_{k}a", "sequence": exact, "isotype": isotype},
                 {"sequence_id": f"v1_{k}b", "sequence": mutated, "isotype": isotype}]
    for k in range(3):
        read = V311[:CYS_END] + _random(21, 200 + k, "ACT") + TAIL
        rows += [{"sequence_id": f"v311_{k}{m}", "sequence": read, "isotype": "IGHG"} for m in "ab"]
    return [{**row, "protocol": "multiplex", "consensus_count": 1, "dupcount": 1, "primer_masked": "[]"}
            for row in rows]


class FakeTool:
    """A tool's runner: it calls each read's source allele (IGHVF1-G1*01 or IGHVF7-G1*01), and ``alignair`` the
    wrong IGHV1-2 allele on every odd clone's mutated read. On a restricted run a call outside the alleles becomes
    IGHVF5-G1*01 (IGHV3-23, not captured). Every run and prepare is recorded."""

    runs: list = []
    prepares: list = []
    version = "1"

    def __init__(self, name: str):
        self.name = name

    def fingerprint(self) -> dict:
        return {"tool": self.name, "version": FakeTool.version}

    def prepare(self, layout, reference) -> dict:
        FakeTool.prepares.append(self.name)
        return {"fake": True}

    def run(self, fasta, out_tsv, *, suite, arm, log_dir, alleles=None) -> RunResult:
        FakeTool.runs.append((self.name, suite, arm))
        allowed = None if alleles is None else set(alleles)
        lines = ["sequence_id\tsequence\tv_call\td_call\tj_call"]
        for read_id, sequence in read_fasta(fasta):
            call = "IGHVF1-G1*01" if read_id.startswith("v1_") else "IGHVF7-G1*01"
            if self.name == "alignair" and read_id.startswith("v1_") and read_id.endswith("b") \
                    and int(read_id[3:-1]) % 2:
                call = "IGHVF1-G1*02"
            if allowed is not None and call not in allowed:
                call = "IGHVF5-G1*01"
            lines.append(f"{read_id}\t{sequence}\t{call}\tIGHD3-10*01\tIGHJ4*02")
        Path(out_tsv).write_text("\n".join(lines) + "\n", encoding="utf-8")
        provenance = {"fake": True}
        if self.name == "alignair-api":
            provenance["sidecar"] = {"counts": {"failed_assemblies": 1, "partial_assemblies": 0,
                                               "complete_assemblies": len(lines) - 2}}
        return RunResult(tool=self.name, suite=suite, arm=arm, predictions=str(out_tsv), n_reads=len(lines) - 1,
                         wall_seconds=0.01, peak_rss_kb=None, returncode=0, commands=[["fake", self.name]],
                         provenance=provenance)


class Fakes:
    """The fake steps, with what each did."""

    def __init__(self, data_root: Path):
        self.data_root = data_root
        self.captures: list[str] = []
        self.preprocessed: list[str] = []
        self.truths: list[str] = []
        self.refuse: set[str] = set()
        self.captured: dict[str, frozenset] = {}
        self.csv_md5 = "csv-md5"

    def fetch_inputs(self, donor, data_root, *, threads=None):
        folder = Path(data_root) / donor
        folder.mkdir(parents=True, exist_ok=True)
        files = {}
        for key in ("vdjbase_csv", "vdjbase_bam", "ogrdb", "gene_coords"):
            files[key] = folder / key
            files[key].write_text(key)
        r1, r2 = folder / "r1.fastq.gz", folder / "r2.fastq.gz"
        r1.write_text("r1")
        r2.write_text("r2")
        md5 = {"reads": "reads-md5", "vdjbase_csv": self.csv_md5, "vdjbase_bam": "bam-md5", "ogrdb": "ogrdb-md5",
               "gene_coords": "bed-md5"}
        record = {"reads": {"md5": "reads-md5", "quality": "full", "detail": {"source": "odp", "run": "SRR0"}},
                  "genotypes": {key: {"md5": value} for key, value in md5.items() if key != "reads"}}
        return real_steps.Fetched(r1=r1, r2=r2, quality="full", files=files, md5=md5, record=record, reused=True)

    def capture_genotype(self, donor, fetched, reference, iuis):
        self.captures.append(donor)
        return _genotype(donor, self.captured.get(donor, frozenset({"IGHV1-2", "IGHV3-11"})))

    def preprocess_run(self, donor, r1, r2, layout, *, quality, cap, paper_length_filter=False,
                       paper_quality_trim=False, accept_assembly_loss=False, nproc=8, timeout=None):
        record_path = layout.real_preprocess_json(donor)
        fingerprint = f"cap={cap} filter={paper_length_filter} accept={accept_assembly_loss}"
        if record_path.is_file():
            old = json.loads(record_path.read_text())
            if old.get("status") == "done" and old.get("fingerprint") == fingerprint:
                return {**old, "reused": True}
        record_path.parent.mkdir(parents=True, exist_ok=True)
        if donor in self.refuse:
            evidence = {"reason": "neither protocol", "race": {"linker": 0.01}, "multiplex": {"vh": 0.2, "c": 0.3}}
            record_path.write_text(json.dumps({"donor": donor, "protocol": "unknown", "status": "refused",
                                               "classifier": {"protocol": "unknown", "evidence": evidence}}))
            raise PreprocessError(f"{donor}: the run is refused: neither protocol", evidence)
        self.preprocessed.append(donor)
        rows = _reads(donor)[:cap]
        layout.real_reads(donor).write_text("".join(f">{r['sequence_id']}\n{r['sequence']}\n" for r in rows))
        with open(layout.real_reads_meta(donor), "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=TSV_COLUMNS, delimiter="\t", extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        record = {"donor": donor, "protocol": "multiplex", "quality": quality, "status": "done",
                  "fingerprint": fingerprint, "accepted": True, "counts": {"raw_pairs": 100, "unique": len(rows),
                                                                           "capped": len(rows)},
                  "classifier": {"protocol": "multiplex", "evidence": {"multiplex": {"vh": 0.95, "c": 0.93}}},
                  "assembly_acceptance": {"failure_rate": 0.16, "accepted": True},
                  "isotypes": dict(Counter(row["isotype"] for row in rows))}
        record_path.write_text(json.dumps(record))
        return {**record, "reused": False}

    def real_truths(self, reads, meta, genotype, reference, **kwargs):
        self.truths.append(genotype.donor)
        return ORIGINAL_TRUTHS(reads, meta, genotype, reference, **kwargs)


ORIGINAL_TRUTHS = real_steps.real_truths
ORIGINAL_CHECK_IMAGES = real_steps.check_images


@pytest.fixture()
def fakes(tmp_path, monkeypatch):
    found = Fakes(tmp_path / "data")
    found.data_root.mkdir()
    FakeTool.runs, FakeTool.prepares, FakeTool.version = [], [], "1"
    for tool in ("igblast", "alignair", "alignair-api"):
        monkeypatch.setitem(RUNNERS, tool, lambda options, tool=tool: FakeTool(tool))
    monkeypatch.setattr(real_steps, "check_images", lambda images: {
        name: {"image": image, "id": f"sha256:{name}", "repo_digests": [image]} for name, image in images.items()})
    monkeypatch.setattr(real_steps, "load_reference", lambda name: REFERENCE)
    monkeypatch.setattr(real_steps, "load_iuis", lambda path, reference: IUIS)
    monkeypatch.setattr(real_steps, "sensitivity_genes", lambda data_root: {
        "genes": ["IGHV3-11"], "sd3_alleles": 2, "missing_alleles": 1, "cohort": {"subjects": 1}, "files": {}})
    monkeypatch.setattr(real_steps, "fetch_manifest", lambda data_root: {"quality": {"SRR0": "full"}})
    monkeypatch.setattr(real_steps, "fetch_inputs", found.fetch_inputs)
    monkeypatch.setattr(real_steps, "capture_genotype", found.capture_genotype)
    monkeypatch.setattr(real_steps, "preprocess_run", found.preprocess_run)
    monkeypatch.setattr(real_steps, "real_truths", found.real_truths)
    return found


def _config(tmp_path: Path, **overrides) -> RealConfig:
    (tmp_path / "igblast").mkdir(exist_ok=True)
    checkpoint = tmp_path / "model.alignair"
    checkpoint.write_bytes(b"weights")
    fields = dict(donors=("W-97",), split="dev", data_root=str(tmp_path / "data"), cap=30, threads=2,
                  igblast_dir=str(tmp_path / "igblast"), alignair_checkpoint=str(checkpoint))
    return RealConfig(**(fields | overrides))


# --- refusals -----------------------------------------------------------------------------------------------------

def test_refuses_test_donors_on_dev(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    with pytest.raises(ValueError, match=r"SC-4 is a test donor.*--split test"):
        run_real(_config(tmp_path, donors=("W-97", "SC-4")), layout)
    # and a dev donor in a test run, or a donor outside the pilot
    with pytest.raises(ValueError, match="W-97 is a dev donor"):
        run_real(_config(tmp_path, donors=("W-97",), split="test"), layout)
    with pytest.raises(ValueError, match=r"unknown donor XX-1.*SC-23"):
        run_real(_config(tmp_path, donors=("XX-1",)), layout)
    assert not layout.root.exists()
    assert fakes.captures == fakes.preprocessed == [] and FakeTool.runs == []
    # asked explicitly, a test donor runs
    summary = run_real(_config(tmp_path, donors=("SC-4",), split="test"), layout)
    assert summary["donors"]["SC-4"]["status"] == "done" and summary["config"]["split"] == "test"


def test_stop_after_preprocess_classifies_every_donor_without_scoring(tmp_path, fakes, monkeypatch, capsys):
    # Nothing is scored, so a test donor runs without --split test, no tool option is needed, and neither are the
    # cohort files of the sensitivity analysis; the split of each donor is recorded.
    def no_cohort(data_root):
        raise AssertionError("the sensitivity analysis's files were read")

    images = []
    monkeypatch.setattr(real_steps, "sensitivity_genes", no_cohort)
    monkeypatch.setattr(real_steps, "check_images", lambda found: images.append(dict(found)) or {
        name: {"image": image, "id": "sha256:x", "repo_digests": [image]} for name, image in found.items()})
    layout = RunLayout(tmp_path / "run")
    cfg = _config(tmp_path, donors=("W-97", "SC-4"), stop_after="preprocess", igblast_dir=None,
                  alignair_checkpoint=None)
    summary = run_real(cfg, layout)
    assert fakes.captures == fakes.preprocessed == ["W-97", "SC-4"]
    assert fakes.truths == [] and FakeTool.runs == [] and FakeTool.prepares == []
    assert set(images[0]) == {"sra_tools", "samtools", "immcantation"}  # no AlignAIR image: no tool runs
    for donor, split in (("W-97", "dev"), ("SC-4", "test")):
        block = summary["donors"][donor]
        assert block["status"] == "done" and block["stopped_after"] == "preprocess" and block["split"] == split
        assert set(block["steps"]) == {"fetch", "genotype", "preprocess"}
        assert block["protocol"] == "multiplex" and block["preprocess"]["classifier"]["protocol"] == "multiplex"
        assert "truth" not in block and "runs" not in block and "comparisons" not in block
        assert not layout.real_truth(donor).exists() and not layout.real_run_dir("igblast", donor, "shared").exists()
    assert summary["pooled"]["donors"] == [] and summary["pooled"]["comparisons"] == []
    assert summary["config"]["stop_after"] == "preprocess" and summary["sensitivity"] is None
    manifest = json.loads(layout.real_manifest().read_text())
    assert manifest["identity"]["stop_after"] == "preprocess" and manifest["identity"]["settings"] == {}
    # a rerun reuses every step
    again = run_real(cfg, layout)["donors"]["SC-4"]["steps"]
    assert {step["status"] for step in again.values()} == {"reused"} and fakes.preprocessed == ["W-97", "SC-4"]
    # through the command line too
    args = ["real", "--run-dir", str(layout.root), "--data-root", cfg.data_root, "--donors", "W-97,SC-4", "--cap",
            "30", "--threads", "2", "--stop-after", "preprocess"]
    capsys.readouterr()
    assert main(args) == 0
    assert "SC-4 preprocess: reused" in capsys.readouterr().out
    # scoring the dev donor in the same run directory is another configuration (--force); its preprocessing is reused
    monkeypatch.setattr(real_steps, "sensitivity_genes", lambda data_root: {
        "genes": [], "sd3_alleles": 0, "missing_alleles": 0, "cohort": {}, "files": {}})
    with pytest.raises(ValueError, match="stop_after preprocess -> None.*--force"):
        run_real(_config(tmp_path), layout)
    summary = run_real(_config(tmp_path), layout, force=True)
    assert summary["donors"]["W-97"]["steps"]["preprocess"]["status"] == "reused"
    assert fakes.preprocessed == ["W-97", "SC-4"] and fakes.truths == ["W-97"]
    # without --stop-after, the test donor is refused as before, and another stop is refused
    with pytest.raises(ValueError, match="--split test"):
        run_real(_config(tmp_path, donors=("SC-4",)), RunLayout(tmp_path / "other"))
    with pytest.raises(ValueError, match="stop_after"):
        run_real(_config(tmp_path, stop_after="truth"), RunLayout(tmp_path / "other"))

def test_tool_refusals_before_any_work(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    with pytest.raises(ValueError, match="--igblast-dir"):
        run_real(_config(tmp_path, igblast_dir=None), layout)
    with pytest.raises(FileNotFoundError, match="nowhere"):
        run_real(_config(tmp_path, igblast_dir=str(tmp_path / "nowhere")), layout)
    with pytest.raises(FileNotFoundError, match="checkpoint"):
        run_real(_config(tmp_path, alignair_checkpoint=str(tmp_path / "missing.alignair")), layout)
    with pytest.raises(ValueError, match="pinned by digest"):
        run_real(_config(tmp_path, alignair_image="alignair/alignair:3.0.0"), layout)
    with pytest.raises(ValueError, match="unknown tool"):
        run_real(_config(tmp_path, tools=("igblast", "alignair-local")), layout)
    # a tool not asked for needs nothing
    run_real(_config(tmp_path, tools=("igblast",), alignair_checkpoint=None), layout)
    assert {tool for tool, _, _ in FakeTool.runs} == {"igblast"}


def test_docker_missing_refuses_before_work(tmp_path, fakes, monkeypatch):
    def no_docker(image, **kwargs):
        raise ContainerError(f"docker is not available, and {image} runs in Docker: install Docker or start its "
                             "daemon")

    monkeypatch.setattr(real_steps, "check_images", ORIGINAL_CHECK_IMAGES)
    monkeypatch.setattr(real_steps, "check_image", no_docker)
    layout = RunLayout(tmp_path / "run")
    with pytest.raises(ContainerError, match="docker is not available"):
        run_real(_config(tmp_path), layout)
    assert not layout.root.exists()
    assert fakes.captures == fakes.preprocessed == [] and FakeTool.runs == [] and FakeTool.prepares == []

    # an image whose digest is not the pinned one is refused the same way
    def other_digest(image, **kwargs):
        raise ContainerError(f"image {image}: its RepoDigests ['x@sha256:0'] lack the pinned digest")

    monkeypatch.setattr(real_steps, "check_image", other_digest)
    with pytest.raises(ContainerError, match="lack the pinned digest"):
        run_real(_config(tmp_path), layout)
    assert not layout.root.exists()


# --- the run ------------------------------------------------------------------------------------------------------

def test_a_run_scores_every_tool_and_arm(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    summary = run_real(_config(tmp_path), layout)
    assert json.loads(layout.real_summary_json().read_text()) == json.loads(json.dumps(summary))
    donor = summary["donors"]["W-97"]
    assert donor["status"] == "done" and donor["protocol"] == "multiplex"
    assert sorted(FakeTool.runs) == sorted([("alignair", "W-97", "shared"), ("alignair-api", "W-97", "genotype"),
                                            ("alignair-api", "W-97", "shared"), ("igblast", "W-97", "genotype"),
                                            ("igblast", "W-97", "shared")])
    # one SupportIndex for every tool: igblast and alignair-api call each read's source allele on the shared arm
    shared = donor["tools"]["shared"]
    assert shared["igblast"]["v_classes"] == {"supported": 24, "unsupported_gene": 6}
    assert shared["alignair"]["v_classes"] == {"supported": 18, "unsupported_allele": 6, "unsupported_gene": 6}
    # restricted to the genotype arm, the IGHV3-11 reads go to IGHV3-23, which is not captured: excluded
    genotype = donor["tools"]["genotype"]
    assert genotype["alignair-api"]["v_classes"] == {"supported": 24, "excluded:not_captured": 6}
    assert genotype["alignair-api"]["exclusions"]["excluded:not_captured"] == 6
    assert donor["runs"]["genotype"]["alignair-api"]["assembly"]["failed_assemblies"] == 1
    # the comparisons, and the arm deltas as information
    pairs = {(c["a"], c["b"], c["arm"]): c for c in donor["comparisons"]}
    assert set(pairs) == {("igblast", "alignair", "shared"), ("igblast", "alignair-api", "shared"),
                          ("igblast", "alignair-api", "genotype")}
    headline = pairs["igblast", "alignair", "shared"]["overall"]["real.v.unsupported"]
    assert headline["n_paired"] == 30 and headline["n_clusters"] == 15 and headline["delta"] == pytest.approx(0.2)
    # on the genotype arm both tools' IGHV3-23 calls take the six reads out of the pair
    genotype_pair = pairs["igblast", "alignair-api", "genotype"]
    assert genotype_pair["excluded"]["v"] == 6
    assert genotype_pair["overall"]["real.v.unsupported"]["n_paired"] == 24
    assert {delta["tool"] for delta in donor["arm_deltas"]} == {"igblast", "alignair-api"}
    assert all(delta["informational"] for delta in donor["arm_deltas"])
    # strata: protocol x isotype
    assert set(pairs["igblast", "alignair", "shared"]["strata"]) == {"multiplex/IGHM", "multiplex/IGHG"}
    # the SD3 sensitivity analysis leaves out the reads of IGHV3-11, an SD3-only gene here
    sensitivity = pairs["igblast", "alignair", "shared"]["sensitivity"]["sd3"]
    assert sensitivity["excluded_reads"] == 6 and sensitivity["genes"] == ["IGHV3-11"]
    assert sensitivity["comparison"]["real.v.unsupported"]["n_paired"] == 24
    # with Mikelov's gene-support thresholds: IGHV1-2 has 12 clonotypes, 6 naive, so nothing more is excluded
    variant = pairs["igblast", "alignair", "shared"]["sensitivity"]["gene_support"]
    assert variant["thresholds"] == {"min_clonotypes": 10, "min_naive_clonotypes": 3}
    assert variant["excluded"]["v"] == 0 and variant["comparison"]["real.v.unsupported"]["n_paired"] == 30
    assert shared["igblast"]["gene_support"]["v_classes"] == shared["igblast"]["v_classes"]
    # the truth's self-check passes, and the pooled summary holds the donor
    assert donor["oracle_self_check"]["passed"] and donor["oracle_self_check"]["reads"] == 12
    assert summary["pooled"]["donors"] == ["W-97"]
    assert {(c["a"], c["b"], c["arm"]) for c in summary["pooled"]["comparisons"]} == set(pairs)
    assert layout.real_comparison_json("W-97", "shared", "igblast_vs_alignair").is_file()
    assert summary["stopped_donors"] == [] and summary["failed_runs"] == []


def test_resume_skips_current_steps(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    cfg = _config(tmp_path)
    first = run_real(cfg, layout)
    assert fakes.captures == fakes.preprocessed == fakes.truths == ["W-97"]
    runs, prepares = list(FakeTool.runs), list(FakeTool.prepares)
    steps = first["donors"]["W-97"]["steps"]
    assert {steps[name]["status"] for name in ("genotype", "preprocess", "truth")} == {"done"}
    assert {step["status"] for step in steps["runs"].values()} == {"done"}

    second = run_real(cfg, layout)
    steps = second["donors"]["W-97"]["steps"]
    assert {steps[name]["status"] for name in ("fetch", "genotype", "preprocess", "truth")} == {"reused"}
    assert {step["status"] for step in steps["runs"].values()} == {"reused"}
    assert {step["status"] for step in steps["scores"].values()} == {"reused"}
    assert fakes.captures == fakes.preprocessed == fakes.truths == ["W-97"]
    assert FakeTool.runs == runs and FakeTool.prepares == prepares  # nothing to run, nothing prepared
    assert second["donors"]["W-97"]["comparisons"] == first["donors"]["W-97"]["comparisons"]

    # a predictions file changed since its run makes that run stale, and only that run
    predictions = layout.real_predictions("alignair", "W-97", "shared")
    predictions.write_text(predictions.read_text().replace("IGHVF1-G1*02", "IGHVF1-G1*01"))
    run_real(cfg, layout)
    assert FakeTool.runs == runs + [("alignair", "W-97", "shared")]


def test_a_changed_genotype_redoes_only_the_genotype_arm(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    cfg = _config(tmp_path)
    run_real(cfg, layout)
    runs = list(FakeTool.runs)
    # another genotype CSV: the capture is redone, and with it the arm's alleles; the shared arm's reads are the same
    fakes.csv_md5 = "another-csv-md5"
    fakes.captured["W-97"] = frozenset({"IGHV1-2"})
    summary = run_real(cfg, layout)
    assert fakes.captures == ["W-97", "W-97"]
    redone = FakeTool.runs[len(runs):]
    assert sorted(redone) == [("alignair-api", "W-97", "genotype"), ("igblast", "W-97", "genotype")]
    steps = summary["donors"]["W-97"]["steps"]
    assert steps["genotype"]["status"] == "done" and steps["truth"]["status"] == "done"
    assert steps["runs"]["shared/igblast"]["status"] == "reused"
    # the arm's identity is in its run record
    record = json.loads(layout.real_run_json("igblast", "W-97", "genotype").read_text())
    identity = record["provenance"]["identity"]["arm"]
    assert set(identity) == {"genotype_sha256", "alleles_sha256", "alleles"}


def test_config_change_needs_force(tmp_path, fakes):
    layout = RunLayout(tmp_path / "run")
    run_real(_config(tmp_path), layout)
    manifest = layout.real_manifest().read_text()
    with pytest.raises(ValueError, match=r"cap 30 -> 20.*--force"):
        run_real(_config(tmp_path, cap=20), layout)
    assert layout.real_manifest().read_text() == manifest
    # a tool changed in place is another configuration too
    FakeTool.version = "2"
    with pytest.raises(ValueError, match=r"igblast\.version 1 -> 2"):
        run_real(_config(tmp_path), layout)
    FakeTool.version = "1"
    summary = run_real(_config(tmp_path, cap=20), layout, force=True)
    replaced = json.loads(layout.real_manifest().read_text())["replaced"]
    assert replaced["changed"] == ["cap"]
    # the preprocessing is redone for the new cap, and the truth and runs with it
    assert fakes.preprocessed == ["W-97", "W-97"] and fakes.truths == ["W-97", "W-97"]
    assert summary["donors"]["W-97"]["truth"]["reads"] == 20
    assert summary["donors"]["W-97"]["steps"]["runs"]["shared/igblast"]["status"] == "done"


def test_unknown_protocol_stops_only_that_donor(tmp_path, fakes):
    fakes.refuse.add("W-133")
    layout = RunLayout(tmp_path / "run")
    summary = run_real(_config(tmp_path, donors=("W-133", "W-97")), layout)
    stopped = summary["donors"]["W-133"]
    assert stopped["status"] == "stopped" and stopped["stopped"]["step"] == "preprocess"
    assert "refused" in stopped["stopped"]["reason"]
    assert stopped["stopped"]["evidence"]["multiplex"] == {"vh": 0.2, "c": 0.3}
    assert stopped["preprocess"]["status"] == "refused"
    assert summary["donors"]["W-97"]["status"] == "done"
    assert {donor for _, donor, _ in FakeTool.runs} == {"W-97"}
    assert summary["pooled"]["donors"] == ["W-97"]
    assert summary["stopped_donors"] == [{"donor": "W-133", "step": "preprocess",
                                          "reason": stopped["stopped"]["reason"]}]


class FailingImage(FakeTool):
    """The released image refusing the job over its AIRR assemblies, as AlignAIR 3.0.0's CLI gate does: its rows are
    written, its return code is 1."""

    def run(self, fasta, out_tsv, **kwargs) -> RunResult:
        result = super().run(fasta, out_tsv, **kwargs)
        error = "AIRR assembly FAILED for 3/30 reads (10.0% > --max-assembly-failures 1.0%)"
        return dataclasses.replace(result, returncode=1, provenance={
            "error": error, "sidecar": {"counts": {"failed_assemblies": 3, "written_records": 30},
                                        "airr_assembly_fail_reasons": {"junction": 3}}})


class SilentTool(FakeTool):
    """A tool that exits 0 without writing its predictions."""

    def run(self, fasta, out_tsv, **kwargs) -> RunResult:
        result = super().run(fasta, out_tsv, **kwargs)
        Path(out_tsv).unlink()
        return result


def test_a_failed_run_is_recorded_and_left_out(tmp_path, fakes, monkeypatch, capsys):
    monkeypatch.setitem(RUNNERS, "alignair", lambda options: FailingImage("alignair"))
    monkeypatch.setitem(RUNNERS, "igblast", lambda options: SilentTool("igblast"))
    layout = RunLayout(tmp_path / "run")
    summary = run_real(_config(tmp_path, tools=("igblast", "alignair", "alignair-api")), layout)
    runs = summary["donors"]["W-97"]["runs"]["shared"]
    assert runs["alignair"]["status"] == "failed" and runs["alignair"]["error"].startswith("AIRR assembly FAILED")
    assert runs["alignair"]["assembly"] == {"written_records": 30, "failed_assemblies": 3,
                                            "fail_reasons": {"junction": 3}}
    assert runs["igblast"]["status"] == "failed" and runs["igblast"]["error"] == "the run wrote no predictions"
    comparisons = summary["donors"]["W-97"]["comparisons"]
    assert {c["status"] for c in comparisons} == {"not compared"}
    assert "alignair" in comparisons[0]["reason"] and summary["pooled"]["comparisons"] == []
    assert {(run["tool"], run["arm"]) for run in summary["failed_runs"]} == {
        ("alignair", "shared"), ("igblast", "shared"), ("igblast", "genotype")}
    assert "alignair" not in summary["donors"]["W-97"]["tools"]["shared"]
    # a failed run is made again on the next invocation
    count = len(FakeTool.runs)
    run_real(_config(tmp_path, tools=("igblast", "alignair", "alignair-api")), layout)
    assert len(FakeTool.runs) == count + 3


def test_low_support_is_a_sensitivity_variant_not_the_primary(tmp_path, fakes):
    # capped to 8 reads, IGHV1-2's known answers make 4 clonotypes: under Mikelov's thresholds (10 clonotypes, 3
    # naive) the gene is low_support. The R1 primary still scores every read, the oracle's among them.
    summary = run_real(_config(tmp_path, cap=8), RunLayout(tmp_path / "run"))
    donor = summary["donors"]["W-97"]
    igblast = donor["tools"]["shared"]["igblast"]
    assert igblast["v_classes"] == {"supported": 8}
    assert igblast["summary"]["real.v.unsupported"]["n"] == 8
    assert igblast["gene_support"]["v_classes"] == {"excluded:low_support": 8}
    assert igblast["gene_support"]["exclusions"]["excluded:low_support"] == 8
    assert "real.v.unsupported" not in igblast["gene_support"]["summary"]
    pair = {(c["a"], c["b"], c["arm"]): c for c in donor["comparisons"]}["igblast", "alignair", "shared"]
    assert pair["overall"]["real.v.unsupported"]["n_paired"] == 8
    variant = pair["sensitivity"]["gene_support"]
    assert variant["excluded"]["v"] == 8 and variant["comparison"] == {}
    assert donor["oracle_self_check"]["status"] == "passed" and donor["oracle_self_check"]["scored"] == 4
    pooled = {(c["a"], c["b"], c["arm"]): c for c in summary["pooled"]["comparisons"]}
    assert pooled["igblast", "alignair", "shared"]["sensitivity"]["gene_support"]["excluded"]["v"] == 8
    assert summary["pooled"]["tools"]["shared"]["igblast"]["gene_support"]["v_classes"] == {
        "excluded:low_support": 8}

def test_the_oracle_self_check_is_judged(tmp_path, fakes, monkeypatch, capsys):
    # on 30 reads every oracle read is scored and supported: passed
    layout = RunLayout(tmp_path / "run")
    summary = run_real(_config(tmp_path), layout)
    assert summary["donors"]["W-97"]["oracle_self_check"]["status"] == "passed"
    # IGHV1-2 not captured: every oracle read is excluded, so the check is unchecked, not failed (the truth is not
    # contradicted), and is listed as such
    fakes.captured["W-97"] = frozenset({"IGHV3-11"})
    summary = run_real(_config(tmp_path), RunLayout(tmp_path / "uncaptured"))
    check = summary["donors"]["W-97"]["oracle_self_check"]
    assert check["scored"] == 0 and check["known_top1"] == 1.0 and check["status"] == "unchecked"
    assert summary["oracle_failures"] == [] and summary["oracle_unchecked"] == ["W-97"]
    cfg = _config(tmp_path)
    args = ["real", "--run-dir", str(tmp_path / "uncaptured"), "--data-root", cfg.data_root, "--donors", "W-97",
            "--cap", "30", "--threads", "2", "--igblast-dir", cfg.igblast_dir, "--alignair-checkpoint",
            cfg.alignair_checkpoint]
    capsys.readouterr()
    assert main(args) == 0
    assert "W-97's truth self-check is unchecked" in capsys.readouterr().err
    # a self-check whose scored oracle reads are not all supported fails, and the command says so
    from alignair_assay.workflow import real as real_module

    monkeypatch.setattr(real_module, "oracle_self_check", lambda truths, index: {
        "reads": 12, "scored": 12, "supported": 11 / 12, "known_top1": 1.0, "known_gene_top1": 1.0, "passed": False})
    assert main(args) == 2
    assert "W-97's truth self-check failed" in capsys.readouterr().err


def test_clusters_passed_to_compare(tmp_path, fakes, monkeypatch):
    calls = []
    original = real_score.compare

    def spy(a, b, metrics, **kwargs):
        calls.append((set(a) | set(b), kwargs.get("clusters")))
        return original(a, b, metrics, **kwargs)

    monkeypatch.setattr(real_score, "compare", spy)
    layout = RunLayout(tmp_path / "run")
    run_real(_config(tmp_path), layout)
    assert calls
    for cases, clusters in calls:
        assert clusters is not None and cases <= set(clusters)
        # every cluster id carries the donor, and the reads of one clone share it
        assert all(cluster.startswith("W-97/") for cluster in clusters.values())
        members = Counter(clusters[case] for case in cases)
        assert max(members.values()) >= 2
    per_donor = [clusters for cases, clusters in calls if "v1_0a" in cases]
    pooled = [clusters for cases, clusters in calls if "W-97/v1_0a" in cases]
    assert per_donor and pooled
    assert per_donor[0]["v1_0a"] == per_donor[0]["v1_0b"] == pooled[0]["W-97/v1_0a"]
    assert per_donor[0]["v1_0a"] != per_donor[0]["v1_1a"]


def test_the_cli_runs_and_refuses(tmp_path, fakes, capsys):
    cfg = _config(tmp_path)
    args = ["real", "--run-dir", str(tmp_path / "run"), "--data-root", cfg.data_root, "--donors", "W-97",
            "--cap", "30", "--threads", "2", "--igblast-dir", cfg.igblast_dir, "--alignair-checkpoint",
            cfg.alignair_checkpoint]
    assert main(args) == 0
    out = capsys.readouterr().out
    assert "W-97 preprocess: done" in out and "summary:" in out
    assert main([*args[:6], "SC-4", *args[7:]]) == 2
    assert "--split test" in capsys.readouterr().err


# --- the real W-97 reads ------------------------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.skipif(not (DATA and DOCKER and MODEL and IGBLAST.is_dir()),
                    reason="set ALIGNAIR_ASSAY_DATA, ALIGNAIR_ASSAY_DOCKER=1 and ALIGNAIR_ASSAY_MODEL (and unpack "
                           "IgBLAST 1.22.0 under .private/tools)")
def test_w97_end_to_end_on_500_reads(tmp_path):
    cfg = RealConfig(donors=("W-97",), split="dev", data_root=DATA, cap=500, threads=8, device="cpu",
                     igblast_dir=str(IGBLAST), alignair_checkpoint=str(Path(MODEL).expanduser()))
    started = time.monotonic()
    summary = run_real(cfg, RunLayout(tmp_path / "run"))
    seconds = time.monotonic() - started
    assert seconds < 600, seconds
    donor = summary["donors"]["W-97"]
    assert donor["status"] == "done", donor.get("stopped")
    # the R1 primary applies no gene-support threshold, so the oracle's reads are scored and all supported even on
    # 500 reads, and the shared arm's calls are scored; the Mikelov variant excludes the thin genes
    assert donor["truth"]["reads"] == 500 and donor["oracle_self_check"]["status"] == "passed"
    assert donor["oracle_self_check"]["known_top1"] == donor["oracle_self_check"]["known_gene_top1"] == 1.0
    igblast = donor["tools"]["shared"]["igblast"]
    assert igblast["summary"]["real.v.unsupported"]["n"] > 100
    assert igblast["gene_support"]["exclusions"]["excluded:low_support"] > 0
    assert "excluded:low_support" not in igblast["v_classes"]
    # every tool and arm ran (on these reads none fails), the AlignAIR runs with their AIRR assembly counts
    assert summary["failed_runs"] == [] and summary["stopped_donors"] == []
    for arm, tools in (("shared", ("igblast", "alignair", "alignair-api")), ("genotype", ("igblast", "alignair-api"))):
        for tool in tools:
            run = donor["runs"][arm][tool]
            assert run["status"] == "ok", (arm, tool, run)
            assert (run["assembly"] is not None) == tool.startswith("alignair"), (arm, tool, run)
    # one SupportIndex: the image and the API, one model through two paths, class every read alike
    assert donor["tools"]["shared"]["alignair"]["v_classes"] == donor["tools"]["shared"]["alignair-api"]["v_classes"]
    assert {c["status"] for c in donor["comparisons"]} == {"compared"} and len(donor["comparisons"]) == 3
    assert summary["pooled"]["donors"] == ["W-97"]
    print(json.dumps({"seconds": round(seconds, 1), "steps": donor["steps"]}, indent=1))
    # a second run reuses every step
    again = run_real(cfg, RunLayout(tmp_path / "run"))["donors"]["W-97"]["steps"]
    assert {again[name]["status"] for name in ("fetch", "genotype", "preprocess", "truth")} == {"reused"}
    assert {step["status"] for kind in ("runs", "scores") for step in again[kind].values()} == {"reused"}
