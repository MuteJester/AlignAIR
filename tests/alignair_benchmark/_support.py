"""Shared deterministic benchmark cases and table writers for the test suite."""
import csv
from alignair_benchmark import (
    BenchmarkCase,
    BenchmarkSpec,
    GeneTruth,
    StratumSpec,
    generate_benchmark,
)


def _small_spec():
    return BenchmarkSpec(
        name="tiny",
        dataconfig_name="HUMAN_IGH_OGRDB",
        seed=7,
        strata=(
            StratumSpec(name="clean", n=3, progress=0.0),
            StratumSpec(name="oriented", n=4, progress=0.3, orientation_ids=(0, 1, 2, 3)),
        ),
    )


def _single_clean_case(seed=31):
    return generate_benchmark(
        BenchmarkSpec(
            name="single",
            dataconfig_name="HUMAN_IGH_OGRDB",
            seed=seed,
            strata=(StratumSpec(name="clean", n=1, progress=0.0),),
        )
    )[0]


def _manual_case(case_id, v_calls, *, stratum="manual", orientation_id=0):
    genes = {
        "v": GeneTruth(
            calls=tuple(v_calls),
            primary=v_calls[0],
            sequence_start=0,
            sequence_end=80,
            germline_start=0,
            germline_end=80,
        ),
        "d": GeneTruth(),
        "j": GeneTruth(
            calls=("IGHJ4*01",),
            primary="IGHJ4*01",
            sequence_start=80,
            sequence_end=100,
            germline_start=0,
            germline_end=20,
        ),
    }
    return BenchmarkCase(
        case_id=case_id,
        stratum=stratum,
        sequence="A" * 100,
        canonical_sequence="A" * 100,
        orientation_id=orientation_id,
        genes=genes,
        presented_genes=genes,
        record={},
    )


class _FakeReferenceGene:
    def __init__(self, names):
        self.names = tuple(names)


class _FakeReferenceSet:
    has_d = True

    def __init__(self):
        self.genes = {
            "V": _FakeReferenceGene(("IGHV1*01", "IGHV2*01")),
            "J": _FakeReferenceGene(("IGHJ4*01",)),
        }

    def gene(self, gene):
        return self.genes[gene]


def _airr_row_for_case(case):
    row = {
        "sequence_id": case.case_id,
        "sequence": case.canonical_sequence,
        "productive": "T" if bool(case.record.get("productive")) else "F",
        "vj_in_frame": "T" if bool(case.record.get("vj_in_frame")) else "F",
        "stop_codon": "T" if bool(case.record.get("stop_codon")) else "F",
        "rev_comp": "F",
    }
    for gene, truth in case.genes.items():
        row[f"{gene}_call"] = ",".join(truth.calls)
        if truth.sequence_start is not None:
            row[f"{gene}_sequence_start"] = str(truth.sequence_start + 1)
        if truth.sequence_end is not None:
            row[f"{gene}_sequence_end"] = str(truth.sequence_end)
        if truth.germline_start is not None:
            row[f"{gene}_germline_start"] = str(truth.germline_start + 1)
        if truth.germline_end is not None:
            row[f"{gene}_germline_end"] = str(truth.germline_end)
    for key in (
        "locus",
        "c_call",
        "junction",
        "junction_aa",
        "junction_length",
        "np1",
        "np2",
        "np1_length",
        "np2_length",
    ):
        if key in case.record:
            row[key] = case.record[key]
    if "junction_start" in case.record:
        row["junction_start"] = str(case.record["junction_start"] + 1)
    if "junction_end" in case.record:
        row["junction_end"] = str(case.record["junction_end"])
    return row


def _write_tsv(path, rows):
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
