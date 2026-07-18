from alignair.data.column_schema import ColumnSet


def test_no_d_columns():
    cs = ColumnSet(has_d=False)
    cols = cs.as_list()
    assert "sequence" in cols and "v_call" in cols and "j_call" in cols
    assert "d_call" not in cols
    assert "v_sequence_start" in cols and "j_sequence_end" in cols


def test_d_columns():
    cs = ColumnSet(has_d=True)
    cols = cs.as_list()
    assert "d_call" in cols and "d_sequence_start" in cols and "d_sequence_end" in cols


def test_label_vs_required():
    cs = ColumnSet(has_d=True)
    # productive/indels are optional labels (defaulted if absent)
    assert "productive" in cs.optional_columns
    assert "indels" in cs.optional_columns
    assert "sequence" in cs.required_columns
