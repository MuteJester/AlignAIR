from alignair.data.record_adapter import RecordAdapter


def _row(**over):
    base = {
        "sequence": "ACGT", "v_call": "V*01", "j_call": "J*01",
        "v_sequence_start": "0", "v_sequence_end": "4",
        "j_sequence_start": "5", "j_sequence_end": "8",
        "mutation_rate": "0.1", "productive": "T", "indels": "{}",
        "d_call": "D*01,D*02", "d_sequence_start": "4", "d_sequence_end": "5",
    }
    base.update(over)
    return base


def test_coord_shift_by_pad():
    ad = RecordAdapter(has_d=True)
    rec = ad.adapt(_row(), pad_left=3)
    assert rec["v_start"] == 3.0 and rec["v_end"] == 7.0
    assert rec["j_start"] == 8.0 and rec["j_end"] == 11.0
    assert rec["d_start"] == 7.0 and rec["d_end"] == 8.0


def test_ambiguous_calls_split_to_set():
    ad = RecordAdapter(has_d=True)
    rec = ad.adapt(_row(), pad_left=0)
    assert rec["d_call_set"] == {"D*01", "D*02"}
    assert rec["v_call_set"] == {"V*01"}


def test_indel_count_from_dict_string():
    ad = RecordAdapter(has_d=False)
    rec = ad.adapt(_row(indels="{'1': 5, '2': 9}"), pad_left=0)
    assert rec["indel_count"] == 2.0


def test_indel_count_empty():
    ad = RecordAdapter(has_d=False)
    assert ad.adapt(_row(indels=""), pad_left=0)["indel_count"] == 0.0


def test_productive_coercion():
    ad = RecordAdapter(has_d=False)
    assert ad.adapt(_row(productive="false"), pad_left=0)["productive"] == 0.0
    assert ad.adapt(_row(productive=1.0), pad_left=0)["productive"] == 1.0
