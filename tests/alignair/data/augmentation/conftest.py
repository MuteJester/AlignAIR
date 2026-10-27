"""Small nucleotide references with independently known protein translations."""
import pytest
from alignair.reference import build_protein_reference
from alignair.reference.reference_set import GeneReference, ReferenceSet
from alignair.data.sampling import ProteinSampleBuilder
from alignair.data.tokenizer import AAProteinTokenizer

_CODONS = dict(zip('ACDEFGHIKLMNPQRSTVWY',
                  ['GCT','TGT','GAT','GAA','TTT','GGT','CAT','ATT','AAA','CTG','ATG','AAT','CCT','CAA','CGT','TCT','ACT','GTT','TGG','TAT']))

@pytest.fixture
def crop_case():
    def make(*, j_frame=0):
        proteins = {'V': {'V1': 'ACDEFGHIKC'},
                    'J': {'J1': 'WGQAA', 'J2': 'WGREE', 'J3': 'AWGQAAV'}}
        genes = {}
        for gene, rows in proteins.items():
            prefix = 'A' * (j_frame if gene == 'J' else 0)
            names = list(rows)
            sequences = [prefix + ''.join(_CODONS[aa] for aa in protein) for protein in rows.values()]
            anchors = {name: len(prefix) + 3 * (protein.index('W') if gene == 'J' else protein.rindex('C'))
                       for name, protein in rows.items()}
            genes[gene] = GeneReference(names, sequences, {name: i for i, name in enumerate(names)}, anchors)
        reference = build_protein_reference(ReferenceSet(genes, has_d=False))
        record = {'sequence': genes['V'].sequences[0] + genes['J'].sequences[0][j_frame:],
                  'sequence_aa': 'ACDEFGHIKCWGQAA', 'junction_start': 0, 'productive': True,
                  'v_call': 'V1', 'j_call': 'J1', 'v_sequence_start': 0, 'v_sequence_end': 30,
                  'v_germline_start': 0, 'v_germline_end': 30, 'v_cigar': '30M',
                  'j_sequence_start': 30, 'j_sequence_end': 45,
                  'j_germline_start': j_frame, 'j_germline_end': j_frame + 15, 'j_cigar': '15M'}
        builder = ProteinSampleBuilder(reference.class_maps, AAProteinTokenizer(), 64)
        return reference, record, builder
    return make
