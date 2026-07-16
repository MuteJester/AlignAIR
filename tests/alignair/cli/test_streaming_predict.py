"""The predict CLI streams reader-chunk -> predict -> assemble -> write in bounded
memory, preserving order + cross-chunk duplicate-id handling, streaming rejects, and accounting by
state. Tested with a fake aligner (no model / no torch needed)."""
import csv

from alignair.cli.predict import _stream_predict


class _Result:
    def __init__(self, recs):
        self._r = recs

    def to_dicts(self):
        return self._r


class _FakeAligner:
    """Returns one 'complete' (or `status`) record per read; counts predict() calls to prove chunking."""
    def __init__(self, status="complete", payload=1):
        self.status = status
        self.payload = payload
        self.calls = 0

    def predict(self, seqs, *, batch_size, airr, **kw):
        self.calls += 1
        return _Result([{"sequence": s, "v_call": "IGHV1-1*01", "airr_assembly_status": self.status,
                         "airr_assembly_reason": ("missing_calls_or_coordinates"
                                                  if self.status == "partial" else None),
                         "big": "A" * self.payload} for s in seqs])


def _rows(path):
    with open(path) as f:
        return list(csv.DictReader(f, delimiter="\t"))


def _fasta(path, n, bad_at=None):
    with open(path, "w") as f:
        for i in range(n):
            f.write(f">r{i}\nACGTACGTACGT\n")
        if bad_at is not None:
            f.write(">bad\nXXXXZZZZ\n")


def test_stream_preserves_order_counts_and_chunks(tmp_path):
    _fasta(tmp_path / "in.fasta", 10, bad_at=True)
    aligner = _FakeAligner()
    counts, fr, pr = _stream_predict(
        aligner, input_path=str(tmp_path / "in.fasta"), out_path=str(tmp_path / "out.tsv"),
        columns="minimal", chunk_size=3, seq_column=None, id_column=None, meta_by_id=None,
        extra_cols=None, out_locus="IGH", overrides={}, batch_size=64,
        rejects_out=str(tmp_path / "rej.tsv"))
    assert aligner.calls == 4                              # ceil(10/3) chunks -> streamed, not one shot
    assert counts == {"input": 11, "accepted": 10, "rejected": 1, "cropped": 0,
                      "complete": 10, "partial": 0, "failed": 0, "written": 10,
                      "nonstandard_orientation": 0}
    rows = _rows(tmp_path / "out.tsv")
    assert [r["sequence_id"] for r in rows] == [f"r{i}" for i in range(10)]   # order preserved
    rej = _rows(tmp_path / "rej.tsv")
    assert len(rej) == 1 and rej[0]["reason"] == "ambiguous"


def test_stream_counts_partial_status(tmp_path):
    _fasta(tmp_path / "in.fasta", 4)
    counts, fr, pr = _stream_predict(
        _FakeAligner(status="partial"), input_path=str(tmp_path / "in.fasta"),
        out_path=str(tmp_path / "out.tsv"), columns="minimal", chunk_size=2, seq_column=None,
        id_column=None, meta_by_id=None, extra_cols=None, out_locus="IGH", overrides={},
        batch_size=64, rejects_out=None)
    assert counts["partial"] == 4 and counts["complete"] == 0
    assert pr == {"missing_calls_or_coordinates": 4}


def test_stream_memory_is_bounded(tmp_path):
    """Peak memory scales with chunk_size, not the repertoire: a large payload held for all records
    would dwarf the streamed peak."""
    import tracemalloc
    n = 40_000
    _fasta(tmp_path / "big.fasta", n)
    aligner = _FakeAligner(payload=800)                   # ~0.8 KB/record -> all-held ~32 MB
    tracemalloc.start()
    counts, _, _ = _stream_predict(
        aligner, input_path=str(tmp_path / "big.fasta"), out_path=str(tmp_path / "out.tsv"),
        columns="minimal", chunk_size=1000, seq_column=None, id_column=None, meta_by_id=None,
        extra_cols=None, out_locus="IGH", overrides={}, batch_size=64, rejects_out=None)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    assert counts["written"] == n
    assert peak < 20_000_000                              # << the ~32 MB an eager (all-records) run needs
