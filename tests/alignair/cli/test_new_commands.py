"""Behavior tests for the new diagnostic/validation/reference commands (no model needed)."""
import argparse

from alignair.cli import doctor, reference, validate_airr as validate_cmd
from alignair.io.airr_validate import cigar_query_length, validate_airr_file


def _run(register, argv, **defaults):
    p = argparse.ArgumentParser()
    sub = p.add_subparsers()
    register(sub)
    args = p.parse_args(argv)
    for k, v in defaults.items():
        setattr(args, k, v)
    return args.func(args)


def test_doctor_reports_core_env_and_exits_zero(capsys):
    info = doctor.diagnostics()
    assert "alignair" in info and "python" in info and "torch" in info
    assert _run(doctor.register, ["doctor"]) == 0
    assert "AlignAIR environment" in capsys.readouterr().out


def test_reference_list_runs(capsys):
    assert _run(reference.register, ["reference", "list", "--species", "Human", "--chain", "IGH"]) == 0
    out = capsys.readouterr().out
    assert "HUMAN_IGH_OGRDB" in out


def test_cigar_query_length_counts_query_ops():
    assert cigar_query_length("5S10M2I3S") == 20          # S+M+I consume query; not D/N
    assert cigar_query_length("3N8M") == 8                # N (germline skip) does not consume query
    assert cigar_query_length("") == 0


def test_validate_airr_flags_out_of_bounds_and_overlong_cigar(tmp_path):
    good = tmp_path / "good.tsv"
    good.write_text("sequence_id\tsequence\tv_call\tj_call\tv_cigar\tv_sequence_start\tv_sequence_end\n"
                    "r1\tACGTACGT\tV1\tJ1\t8M\t1\t8\n")
    rep = validate_airr_file(str(good))
    assert rep["errors"] == [] and rep["n_rows"] == 1

    bad = tmp_path / "bad.tsv"
    bad.write_text("sequence_id\tsequence\tv_call\tj_call\tv_cigar\tv_sequence_start\tv_sequence_end\n"
                   "r1\tACGT\tV1\tJ1\t99M\t3\t2\n")               # cigar over-consumes; start>end
    rep = validate_airr_file(str(bad))
    assert len(rep["errors"]) == 2


def test_validate_airr_missing_columns(tmp_path):
    p = tmp_path / "m.tsv"
    p.write_text("sequence_id\tsequence\n" "r1\tACGT\n")          # no v_call/j_call
    rep = validate_airr_file(str(p))
    assert "v_call" in rep["missing_columns"] and "j_call" in rep["missing_columns"]


def test_validate_cmd_returns_nonzero_on_invalid(tmp_path):
    p = tmp_path / "bad.tsv"
    p.write_text("sequence_id\tsequence\tv_call\tj_call\tv_sequence_start\tv_sequence_end\n"
                 "r1\tACGT\tV1\tJ1\t3\t99\n")
    assert _run(validate_cmd.register, ["validate-airr", str(p)], max_show=20) == 1
