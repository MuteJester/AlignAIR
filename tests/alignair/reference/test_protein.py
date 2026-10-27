from copy import deepcopy

import pytest

from alignair.errors import ConfigurationError, DataContractError
from alignair.reference import GeneReference, ReferenceSet, build_protein_reference, build_protein_germlines
from alignair.reference.labels import db_hash

from alignair.reference.protein import (
    DEFAULT_ANCHOR_RESIDUES, ProteinGermline, translate_germline,
)

pytestmark = pytest.mark.component


def test_translate_germline_passes_anchor_qc():
    # frame 0, Cys (TGT) codon at nt index 6 -> anchor 6, residue index 2.
    seq = "ATGAAATGTGGG"          # M K C G
    germ, reason = translate_germline("IGHV-test*01", seq, anchor=6, gene="V")
    assert reason is None
    assert isinstance(germ, ProteinGermline)
    assert germ.aa_seq == "MKCG"
    assert germ.frame == 0
    assert germ.anchor_aa == 2
    assert germ.aa_seq[germ.anchor_aa] == DEFAULT_ANCHOR_RESIDUES["V"]


def test_translate_germline_quarantines_wrong_anchor_residue():
    # anchor points at GGG (Gly), not Cys -> quarantined.
    seq = "ATGAAAGGGTGG"          # M K G W
    germ, reason = translate_germline("IGHV-bad*01", seq, anchor=6, gene="V")
    assert germ is None
    assert reason is not None and "C" in reason


def test_translate_germline_respects_nonzero_frame():
    # frame 2 (anchor 2 -> 2 % 3 == 2). Translate from base 2: TGT GGG AAA -> C G K ; anchor residue 0.
    seq2 = "AATGTGGGAAA"          # bases[2:] = TGTGGGAAA -> C G K ; anchor 2 -> residue 0
    germ2, reason2 = translate_germline("IGHV-f2b*01", seq2, anchor=2, gene="V")
    assert reason2 is None and germ2.frame == 2 and germ2.anchor_aa == 0
    assert germ2.aa_seq[0] == "C"


def _reference():
    names = {'V': ['V3', 'V1', 'V2', 'Vbad', 'Vmissing'], 'J': ['J2', 'J1'], 'D': ['D2', 'D1']}
    sequences = {'V': ['ATGTGTAAA', 'ATGTGCAAA', 'ATGTGTTTT', 'ATGGGTAAA', 'ATGTGTAAA'],
                 'J': ['TGGAAA', 'TGGTTT'], 'D': ['ACG', 'TAC']}
    anchors = {'V': {name: 3 for name in names['V'][:-1]}, 'J': {'J2': 0, 'J1': 0}, 'D': None}
    return ReferenceSet({gene: GeneReference(order, sequences[gene], {name: i for i, name in enumerate(order)}, anchors[gene])
                         for gene, order in names.items()}, has_d=True)


def test_protein_view_keeps_source_order_and_uses_deterministic_protein_class_ids():
    source = _reference()
    before = deepcopy(source.to_serializable())
    view = build_protein_reference(source)
    assert view.source is source and source.to_serializable() == before
    assert source.gene('V').names == ['V3', 'V1', 'V2', 'Vbad', 'Vmissing']
    assert view.class_maps['V'].allele_class == {'V1': 0, 'V2': 1, 'V3': 0}
    assert view.class_maps['V'].class_members == {0: ['V1', 'V3'], 1: ['V2']}
    assert view.class_maps['J'].allele_class == {'J1': 0, 'J2': 1}
    assert view.source.gene('D').index == {'D2': 0, 'D1': 1}
    assert 'D' not in view.class_maps
    assert view.germlines.quarantined['V'] == {'Vbad': "anchor residue 'G' != expected 'C'", 'Vmissing': 'no conserved anchor'}
    assert view.metadata() == {'n_v_classes': 2, 'n_j_classes': 2, 'db_hash': view.db_hash,
                               'quarantined_v': 2, 'quarantined_j': 0}


def test_hash_scope_is_explicit_and_includes_quarantined_source_alleles():
    source = _reference()
    vj = build_protein_reference(source)
    vdj = build_protein_reference(source, hash_genes=('V', 'D', 'J'))
    expected = db_hash((name, seq) for gene in ('V', 'J') for name, seq in zip(source.gene(gene).names, source.gene(gene).sequences))
    assert vj.hash_genes == ('V', 'J') and vj.db_hash == expected
    assert vdj.db_hash != vj.db_hash
    assert vdj.class_maps['V'].allele_class == vj.class_maps['V'].allele_class
    assert vdj.class_maps['V'].db_hash == vdj.db_hash
    with pytest.raises(ConfigurationError, match='every projected gene'):
        build_protein_reference(source, hash_genes=('D',))


@pytest.mark.parametrize('anchor', [-1, -3, True, 1.5])
def test_invalid_anchors_are_quarantined_instead_of_indexing_an_unrelated_residue(anchor):
    row, reason = translate_germline('bad', 'ATGTGT', anchor, 'V')
    assert row is None and reason == 'invalid conserved anchor'


def test_out_of_range_anchors_and_explicit_residue_policies():
    row, reason = translate_germline('bad', 'ATGTGT', 12, 'V')
    assert row is None and 'out of range' in reason
    source = _reference()
    source.gene('J').sequences = ['TTTAAA', 'TTTTTT']
    with pytest.raises(DataContractError, match='no usable protein labels') as caught:
        build_protein_reference(source)
    assert caught.value.context['gene'] == 'J'
    view = build_protein_reference(source, anchor_residues={'J': 'F'})
    assert {row.aa_seq for row in view.germlines.genes['J'].values()} == {'FK', 'FF'}


@pytest.mark.parametrize('genes', [('D',), (), ('V', 'V'), ('X',), None])
def test_invalid_projection_gene_policies_fail_explicitly(genes):
    with pytest.raises(ConfigurationError):
        build_protein_germlines(_reference(), genes)


@pytest.mark.parametrize('policy', [{'J': 'X'}, {'D': 'C'}, {'J': ''}, {'J': 'CW'}, []])
def test_invalid_anchor_policies_are_configuration_errors(policy):
    with pytest.raises(ConfigurationError):
        build_protein_germlines(_reference(), anchor_residues=policy)


@pytest.mark.parametrize('defect', ['order', 'length', 'duplicates', 'missing_gene', 'bool_index',
                                   'float_index', 'empty_name', 'nonstring_sequence'])
def test_reference_view_refuses_inconsistent_source_vocabularies(defect):
    source = _reference()
    if defect == 'order': source.gene('V').index['V1'] = 0
    elif defect == 'length': source.gene('V').sequences.pop()
    elif defect == 'duplicates': source.gene('V').names[0] = 'V1'
    elif defect == 'bool_index': source.gene('V').index['V3'] = False
    elif defect == 'float_index': source.gene('V').index['V3'] = 0.0
    elif defect == 'empty_name': source.gene('V').names[0] = ''
    elif defect == 'nonstring_sequence': source.gene('V').sequences[0] = None
    else: del source.genes['V']
    with pytest.raises(ConfigurationError if defect == 'missing_gene' else DataContractError):
        build_protein_reference(source)


def test_invalid_sources_and_old_imports_have_clear_contracts():
    from alignair.aa.reference import AAGermline, build_aa_germlines
    from alignair.aa.data import build_class_maps as old_builder
    from alignair.reference.protein import build_class_maps
    assert AAGermline is ProteinGermline and build_aa_germlines is build_protein_germlines
    assert old_builder is build_class_maps
    for source in (None, 'NOT_A_REFERENCE'):
        with pytest.raises(ConfigurationError): build_protein_reference(source)
