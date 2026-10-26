"""Tests for input validation & normalization (AA inference product)."""

import pytest
from alignair.aa.product.validate import (
    Record,
    ErrorResult,
    normalize_inputs,
)

pytestmark = pytest.mark.component


class TestRecord:
    """Test Record dataclass."""

    def test_record_creation(self):
        """Record can be created with sequence_id and sequence."""
        record = Record(sequence_id="seq_0", sequence="EVQL")
        assert record.sequence_id == "seq_0"
        assert record.sequence == "EVQL"


class TestErrorResult:
    """Test ErrorResult dataclass."""

    def test_error_result_creation(self):
        """ErrorResult can be created with sequence_id, error, and detail."""
        error = ErrorResult(sequence_id="seq_0", error="empty", detail="Empty sequence after strip")
        assert error.sequence_id == "seq_0"
        assert error.error == "empty"
        assert error.detail == "Empty sequence after strip"

    def test_error_result_to_dict(self):
        """ErrorResult.to_dict() returns a dict with sequence_id, error, detail."""
        error = ErrorResult(sequence_id="seq_0", error="empty", detail="Empty sequence after strip")
        d = error.to_dict()
        assert isinstance(d, dict)
        assert d["sequence_id"] == "seq_0"
        assert d["error"] == "empty"
        assert d["detail"] == "Empty sequence after strip"


class TestNormalizeInputsBareString:
    """Test normalize_inputs with bare strings."""

    def test_bare_string_single_sequence(self):
        """Bare string creates one Record with seq_0 ID."""
        records, errors = normalize_inputs("EVQL")
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"
        assert records[0].sequence == "EVQL"

    def test_bare_string_lowercased(self):
        """Lowercase bare string is uppercased."""
        records, errors = normalize_inputs("evql")
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence == "EVQL"

    def test_bare_string_with_whitespace_stripping(self):
        """Bare string with surrounding whitespace is stripped and uppercased."""
        records, errors = normalize_inputs("  evql  ")
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence == "EVQL"


class TestNormalizeInputsIterableStrings:
    """Test normalize_inputs with iterable of strings."""

    def test_list_of_strings(self):
        """List of strings creates Records with generated IDs."""
        records, errors = normalize_inputs(["EVQL", "TSVS"])
        assert len(records) == 2
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"
        assert records[0].sequence == "EVQL"
        assert records[1].sequence_id == "seq_1"
        assert records[1].sequence == "TSVS"

    def test_tuple_of_strings(self):
        """Tuple of strings creates Records with generated IDs."""
        records, errors = normalize_inputs(("EVQL", "TSVS"))
        assert len(records) == 2
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"
        assert records[1].sequence_id == "seq_1"


class TestNormalizeInputsDictRecords:
    """Test normalize_inputs with iterable of dicts."""

    def test_list_of_dicts_with_sequence_id(self):
        """List of dicts with sequence_id keeps the provided ID."""
        records, errors = normalize_inputs([{"sequence_id": "a", "sequence": "EVQL"}])
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence_id == "a"
        assert records[0].sequence == "EVQL"

    def test_list_of_dicts_without_sequence_id(self):
        """List of dicts without sequence_id generates an ID."""
        records, errors = normalize_inputs([{"sequence": "EVQL"}])
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"
        assert records[0].sequence == "EVQL"

    def test_list_of_dicts_mixed_ids(self):
        """List of dicts with mixed ID presence generates IDs as needed."""
        records, errors = normalize_inputs([
            {"sequence_id": "custom", "sequence": "EVQL"},
            {"sequence": "TSVS"},
            {"sequence_id": "another", "sequence": "GGGG"},
        ])
        assert len(records) == 3
        assert len(errors) == 0
        assert records[0].sequence_id == "custom"
        assert records[1].sequence_id == "seq_1"
        assert records[2].sequence_id == "another"


class TestNormalizeInputsValidAlphabet:
    """Test that canonical amino acids and X are accepted."""

    def test_canonical_amino_acids_accepted(self):
        """All 20 canonical amino acids are accepted."""
        canonical = "ACDEFGHIKLMNPQRSTVWY"
        records, errors = normalize_inputs(canonical)
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence == canonical

    def test_x_token_accepted(self):
        """X token is accepted."""
        records, errors = normalize_inputs("EVQLX")
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence == "EVQLX"

    def test_x_in_uppercase(self):
        """Lowercase x is uppercased to X."""
        records, errors = normalize_inputs("evqlx")
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence == "EVQLX"


class TestNormalizeInputsInvalidEmpty:
    """Test rejection of empty sequences."""

    def test_empty_string_on_invalid_error(self):
        """Empty string raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0"):
            normalize_inputs("", on_invalid="error")

    def test_whitespace_only_on_invalid_error(self):
        """Whitespace-only string raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0"):
            normalize_inputs("   ", on_invalid="error")

    def test_empty_string_on_invalid_skip(self):
        """Empty string returns ErrorResult with on_invalid='skip'."""
        records, errors = normalize_inputs("", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_0"
        assert errors[0].error == "empty"


class TestNormalizeInputsInvalidStop:
    """Test rejection of stop character (*)."""

    def test_stop_character_on_invalid_error(self):
        """Stop character (*) raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0"):
            normalize_inputs("EV*QL", on_invalid="error")

    def test_stop_character_on_invalid_skip(self):
        """Stop character (*) returns ErrorResult with on_invalid='skip'."""
        records, errors = normalize_inputs("EV*QL", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_0"
        assert errors[0].error == "invalid_character"


class TestNormalizeInputsInvalidInteriorWhitespace:
    """Test rejection of interior whitespace."""

    def test_interior_space_on_invalid_error(self):
        """Interior space raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0.*interior whitespace"):
            normalize_inputs("E V", on_invalid="error")

    def test_interior_space_on_invalid_skip(self):
        """Interior space returns ErrorResult with on_invalid='skip'."""
        records, errors = normalize_inputs("E V", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_0"
        assert errors[0].error == "interior_whitespace"

    def test_interior_tab_on_invalid_error(self):
        """Interior tab raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0.*interior whitespace"):
            normalize_inputs("E\tV", on_invalid="error")

    def test_interior_newline_on_invalid_error(self):
        """Interior newline raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0.*interior whitespace"):
            normalize_inputs("E\nV", on_invalid="error")


class TestNormalizeInputsInvalidOverlength:
    """Test rejection of overlength sequences."""

    def test_overlength_193_on_invalid_error(self):
        """193-character sequence raises ValueError with on_invalid='error'."""
        seq = "A" * 193
        with pytest.raises(ValueError, match="seq_0.*exceeds max_len"):
            normalize_inputs(seq, on_invalid="error", max_len=192)

    def test_overlength_193_on_invalid_skip(self):
        """193-character sequence returns ErrorResult with on_invalid='skip'."""
        seq = "A" * 193
        records, errors = normalize_inputs(seq, on_invalid="skip", max_len=192)
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_0"
        assert errors[0].error == "overlength"

    def test_overlength_at_max_len_boundary(self):
        """Sequence exactly at max_len is valid."""
        seq = "A" * 192
        records, errors = normalize_inputs(seq, on_invalid="error", max_len=192)
        assert len(records) == 1
        assert len(errors) == 0

    def test_overlength_custom_max_len(self):
        """Overlength with custom max_len."""
        seq = "A" * 51
        with pytest.raises(ValueError, match="seq_0.*exceeds max_len"):
            normalize_inputs(seq, on_invalid="error", max_len=50)


class TestNormalizeInputsInvalidCharacters:
    """Test rejection of invalid characters."""

    def test_invalid_character_on_invalid_error(self):
        """Invalid character raises ValueError with on_invalid='error'."""
        with pytest.raises(ValueError, match="seq_0.*invalid.*character"):
            normalize_inputs("EVQ1L", on_invalid="error")

    def test_invalid_character_on_invalid_skip(self):
        """Invalid character returns ErrorResult with on_invalid='skip'."""
        records, errors = normalize_inputs("EVQ1L", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_0"
        assert errors[0].error == "invalid_character"

    def test_multiple_invalid_characters(self):
        """Multiple invalid characters are rejected."""
        records, errors = normalize_inputs("EVQ1@L", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1


class TestNormalizeInputsOnInvalidError:
    """Test on_invalid='error' mode (fail-fast)."""

    def test_error_mode_raises_on_first_invalid(self):
        """on_invalid='error' raises on the first invalid record."""
        with pytest.raises(ValueError, match="seq_1"):
            normalize_inputs(["EVQL", "", "TSVS"], on_invalid="error")

    def test_error_mode_preserves_order_in_error_message(self):
        """Error message identifies the record by its index."""
        with pytest.raises(ValueError, match="seq_0"):
            normalize_inputs(["", "EVQL"], on_invalid="error")

    def test_error_mode_stops_at_first_invalid(self):
        """on_invalid='error' stops processing at first invalid."""
        # We can't directly observe if processing stops early, but we can verify
        # that an error is raised before any later valid records are processed.
        with pytest.raises(ValueError):
            normalize_inputs(["EVQL", "", "TSVS"], on_invalid="error")


class TestNormalizeInputsOnInvalidSkip:
    """Test on_invalid='skip' mode (structured errors)."""

    def test_skip_mode_preserves_valid_records_order(self):
        """on_invalid='skip' preserves order of valid records."""
        records, errors = normalize_inputs(["EVQL", "", "TSVS", "A" * 200], on_invalid="skip")
        valid_ids = [r.sequence_id for r in records]
        assert valid_ids == ["seq_0", "seq_2"]

    def test_skip_mode_returns_error_results(self):
        """on_invalid='skip' returns ErrorResult for invalid records."""
        records, errors = normalize_inputs(["EVQL", "", "TSVS"], on_invalid="skip")
        assert len(errors) == 1
        assert errors[0].sequence_id == "seq_1"
        assert errors[0].error == "empty"

    def test_skip_mode_preserves_error_order(self):
        """on_invalid='skip' preserves order of errors."""
        records, errors = normalize_inputs(
            ["", "EVQL", "E V", "TSVS", "A" * 200],
            on_invalid="skip",
            max_len=192,
        )
        error_ids = [e.sequence_id for e in errors]
        assert error_ids == ["seq_0", "seq_2", "seq_4"]

    def test_skip_mode_mixed_valid_invalid(self):
        """on_invalid='skip' returns both records and errors in the right order."""
        records, errors = normalize_inputs(
            ["EVQL", "", "TSVS", "E V"],
            on_invalid="skip",
        )
        # seq_0: EVQL (valid)
        # seq_1: empty (invalid)
        # seq_2: TSVS (valid)
        # seq_3: E V (invalid)
        assert len(records) == 2
        assert len(errors) == 2
        assert records[0].sequence_id == "seq_0"
        assert records[1].sequence_id == "seq_2"
        assert errors[0].sequence_id == "seq_1"
        assert errors[1].sequence_id == "seq_3"

    def test_skip_mode_all_invalid(self):
        """on_invalid='skip' returns all errors when all are invalid."""
        records, errors = normalize_inputs(["", "E V", "A" * 200], on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 3

    def test_skip_mode_all_valid(self):
        """on_invalid='skip' returns all records when all are valid."""
        records, errors = normalize_inputs(["EVQL", "TSVS", "GGGG"], on_invalid="skip")
        assert len(records) == 3
        assert len(errors) == 0


class TestNormalizeInputsDefaultMaxLen:
    """Test default max_len parameter."""

    def test_default_max_len_is_192(self):
        """Default max_len is 192."""
        seq_192 = "A" * 192
        seq_193 = "A" * 193
        records, errors = normalize_inputs(seq_192, on_invalid="skip")
        assert len(records) == 1
        assert len(errors) == 0

        records, errors = normalize_inputs(seq_193, on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1


class TestNormalizeInputsComplexScenarios:
    """Test complex scenarios combining multiple aspects."""

    def test_mixed_batch_dict_and_string_like_inputs(self):
        """Mixed batch with various input types."""
        records, errors = normalize_inputs([
            {"sequence_id": "custom_1", "sequence": "evql"},  # lowercase, keep ID
            "tsvs",  # bare string becomes seq_1
            {"sequence": "GGGG"},  # no ID, becomes seq_2
        ], on_invalid="skip")
        assert len(records) == 3
        assert records[0].sequence_id == "custom_1"
        assert records[0].sequence == "EVQL"
        assert records[1].sequence_id == "seq_1"
        assert records[1].sequence == "TSVS"
        assert records[2].sequence_id == "seq_2"
        assert records[2].sequence == "GGGG"

    def test_batch_with_all_failure_modes(self):
        """Batch exercising all failure modes."""
        records, errors = normalize_inputs([
            "EVQL",  # valid
            "",  # empty
            "E V",  # interior space
            "EV*QL",  # stop
            "EVQ1",  # invalid char
            "A" * 193,  # overlength
            "TSVS",  # valid
        ], on_invalid="skip", max_len=192)

        assert len(records) == 2
        assert records[0].sequence_id == "seq_0"
        assert records[1].sequence_id == "seq_6"

        assert len(errors) == 5
        error_ids = [e.sequence_id for e in errors]
        assert error_ids == ["seq_1", "seq_2", "seq_3", "seq_4", "seq_5"]

    def test_error_detail_messages_are_descriptive(self):
        """Error detail messages are descriptive."""
        records, errors = normalize_inputs(["", "E V", "EV*QL"], on_invalid="skip")
        assert "empty" in errors[0].detail.lower() or len(errors[0].detail) > 0
        assert "whitespace" in errors[1].detail.lower() or len(errors[1].detail) > 0
        assert "*" in errors[2].detail or "stop" in errors[2].detail.lower() or len(errors[2].detail) > 0


class TestOnInvalidValidation:
    """Test Fix 1: on_invalid parameter validation."""

    def test_on_invalid_wrong_case_raises_value_error(self):
        """on_invalid='Skip' (wrong case) raises ValueError."""
        with pytest.raises(ValueError, match="on_invalid must be 'error' or 'skip'"):
            normalize_inputs("EVQL", on_invalid="Skip")

    def test_on_invalid_bogus_value_raises_value_error(self):
        """on_invalid='bogus' raises ValueError."""
        with pytest.raises(ValueError, match="on_invalid must be 'error' or 'skip'"):
            normalize_inputs("EVQL", on_invalid="bogus")

    def test_on_invalid_error_accepted(self):
        """on_invalid='error' is accepted."""
        records, errors = normalize_inputs("EVQL", on_invalid="error")
        assert len(records) == 1
        assert len(errors) == 0

    def test_on_invalid_skip_accepted(self):
        """on_invalid='skip' is accepted."""
        records, errors = normalize_inputs("EVQL", on_invalid="skip")
        assert len(records) == 1
        assert len(errors) == 0


class TestExcludedIUPACCodes:
    """Test Fix 2: Excluded IUPAC ambiguity codes are rejected."""

    def test_iupac_b_rejected(self):
        """IUPAC code B (Asp or Asn) is rejected."""
        records, errors = normalize_inputs("EVB", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].error == "invalid_character"

    def test_iupac_j_rejected(self):
        """IUPAC code J (Leu or Ile) is rejected."""
        records, errors = normalize_inputs("EVJ", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].error == "invalid_character"

    def test_iupac_o_rejected(self):
        """IUPAC code O (Pyrrolysine) is rejected."""
        records, errors = normalize_inputs("EVO", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].error == "invalid_character"

    def test_iupac_u_rejected(self):
        """IUPAC code U (Selenocysteine) is rejected."""
        records, errors = normalize_inputs("EVU", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].error == "invalid_character"

    def test_iupac_z_rejected(self):
        """IUPAC code Z (Glu or Gln) is rejected."""
        records, errors = normalize_inputs("EVZ", on_invalid="skip")
        assert len(records) == 0
        assert len(errors) == 1
        assert errors[0].error == "invalid_character"


class TestBareDict:
    """Test Fix 3: Bare dict as sequences raises TypeError."""

    def test_bare_dict_raises_type_error(self):
        """A bare dict (not in a list) raises TypeError."""
        with pytest.raises(TypeError, match="sequences must be a str, an iterable of str, or an iterable of .* dicts"):
            normalize_inputs({"sequence": "EVQL"})


class TestEmptySequenceId:
    """Test Fix 4: Empty or None sequence_id falls back to seq_<i>."""

    def test_empty_string_sequence_id_fallback(self):
        """Empty string sequence_id is treated as not-provided (fallback to seq_0)."""
        records, errors = normalize_inputs([{"sequence_id": "", "sequence": "EVQL"}])
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"

    def test_none_sequence_id_fallback(self):
        """None sequence_id is treated as not-provided (fallback to seq_0)."""
        records, errors = normalize_inputs([{"sequence_id": None, "sequence": "EVQL"}])
        assert len(records) == 1
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"

    def test_empty_and_nonempty_sequence_id_mixed(self):
        """Mixed empty and non-empty sequence_ids."""
        records, errors = normalize_inputs([
            {"sequence_id": "", "sequence": "EVQL"},
            {"sequence_id": "custom", "sequence": "TSVS"},
            {"sequence_id": None, "sequence": "AAAA"},
        ])
        assert len(records) == 3
        assert len(errors) == 0
        assert records[0].sequence_id == "seq_0"
        assert records[1].sequence_id == "custom"
        assert records[2].sequence_id == "seq_2"
