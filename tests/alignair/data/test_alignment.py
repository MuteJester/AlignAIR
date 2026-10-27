import pytest
from alignair.data.alignment import parse_cigar, query_reference_map
from alignair.errors import TrainingDataError

pytestmark = pytest.mark.component

@pytest.mark.parametrize('cigar', [None, '', '*'])
def test_unknown_alignment_has_no_invented_offset(cigar):
    assert parse_cigar(cigar) == ()
    result = query_reference_map(cigar, (20, 23), (5, 8))
    assert result.reason == 'missing_cigar' and result.positions == ()

@pytest.mark.parametrize('cigar', ['0M', '2Mgarbage', 'M', '10', '2M0I', '2Q', 2, False, []])
def test_malformed_alignment_raises(cigar):
    with pytest.raises(TrainingDataError): parse_cigar(cigar)


def test_operations_consume_the_correct_coordinate_axes():
    result = query_reference_map('1H1S2M1I1=1X2D1N1M1P', (20, 27), (5, 13))
    assert result.reason is None
    assert result.positions == (None, 5, 6, None, 7, 8, 12)

@pytest.mark.parametrize('query,reference', [((0, 4), (0, 3)), ((0, 3), (0, 4))])
def test_path_disagreement_is_unavailable(query, reference):
    result = query_reference_map('3M', query, reference)
    assert result.reason == 'alignment_span_mismatch' and not result.positions

@pytest.mark.parametrize('query,reference', [((-1, 2),(0,3)), ((3,2),(0,3)), ((0,3),(4,2)),
    ((True,3),(0,3)), ((0,3),(0,1.5)), ((0,),(0,3)), (None,(0,3))])
def test_invalid_coordinate_contract_raises(query, reference):
    with pytest.raises(TrainingDataError): query_reference_map('3M', query, reference)


def test_internal_garbage_in_cigar_is_rejected():
    with pytest.raises(TrainingDataError):parse_cigar('3Mbad4M')
