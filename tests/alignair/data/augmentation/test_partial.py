from copy import deepcopy
from dataclasses import replace
import numpy as np
import pytest

from alignair.data.augmentation import (ProteinCropAugmenter, ProteinCropPolicy,
    crop_protein_sample, reference_residue_evidence, fragment_compatible_classes)
from alignair.errors import TrainingDataError, ConfigurationError

pytestmark = pytest.mark.component


def test_j_read_offset_uses_germline_fragments_instead_of_empty_prototype_slices(crop_case):
    ref, record, builder = crop_case()
    sample = crop_protein_sample(record, builder(record).sample, ref, 10, 13)
    # Read residues 10:13 are germline residues 0:3, not prototype[10:13] == ''.
    assert (sample['j_start'], sample['j_end']) == (0, 3)
    expected = {ref.class_maps['J'].class_of(name) for name in ('J1', 'J3')}
    assert sample['j_compatible'] == expected
    assert sample['j_class_supervise'] and sample['j_reference_evidence'] == 3
    assert sample['v_valid'] == 0 and sample['v_class'] == -1
    assert not sample['v_compatible']


def test_trimmed_j_and_distinct_query_reference_frames(crop_case):
    ref, record, _ = crop_case(j_frame=2)
    # Query codons begin at nt 31; reference coding frame begins at nt 2.
    record.update(sequence='A' + record['sequence'], junction_start=1,
                  v_sequence_start=1, v_sequence_end=31, j_sequence_start=31, j_sequence_end=43,
                  j_germline_start=5, j_germline_end=17, j_cigar='12M')
    evidence = reference_residue_evidence(record, 'j', ref, 10, 14)
    assert evidence.fragments == ('GQAA',) and evidence.positions == (1, 2, 3, 4)


@pytest.mark.parametrize('cigar,query_length,reference_length,fragments,positions', [
    ('3M3I6M',12,9,('W','GQ'),(0,1,2)),
    ('3M3D6M',9,12,('W','QA'),(0,2,3)),
    ('3M1I8M',12,11,('W',),(0,)),
    ('1M1I2M1D8M',12,12,('QA',),(2,3)),
    ('1S12M',13,12,(),()),
])
def test_only_consecutive_reference_codon_bases_supply_evidence(crop_case, cigar, query_length,
                                                              reference_length, fragments, positions):
    ref, record, _ = crop_case()
    record.update(j_cigar=cigar, j_sequence_end=30+query_length, j_germline_end=reference_length)
    evidence = reference_residue_evidence(record, 'J', ref, 10, 15)
    assert evidence.fragments == fragments and evidence.positions == positions


@pytest.mark.parametrize('changes,reason', [
    ({'j_cigar':None}, 'missing_cigar'),
    ({'j_germline_start':None}, 'missing_reference_coordinates'),
    ({'j_cigar':'12M'}, 'alignment_span_mismatch'),
    ({'j_germline_start':1,'j_germline_end':16}, 'reference_span_out_of_range'),
    ({'j_sequence_end':42,'j_germline_start':1,'j_germline_end':13,'j_cigar':'12M'}, 'no_reference_codon_evidence'),
])
def test_unavailable_class_evidence_keeps_known_boundary_supervision(crop_case,changes,reason):
    ref, record, builder = crop_case()
    record.update(changes)
    sample = crop_protein_sample(record, builder(record).sample, ref, 10, 13)
    assert sample['j_valid'] == 1 and sample['j_end'] == 3
    assert not sample['j_class_supervise'] and not sample['j_compatible']
    assert sample['j_class'] == -1 and sample['j_class_reason'] == reason


def test_shm_residues_do_not_exclude_the_source_germline_class(crop_case):
    ref, record, builder = crop_case()
    record['sequence_aa'] = record['sequence_aa'][:10] + 'REEEE'
    sample = crop_protein_sample(record, builder(record).sample, ref, 10, 13)
    assert ref.class_maps['J'].class_of('J1') in sample['j_compatible']
    assert ref.class_maps['J'].class_of('J2') not in sample['j_compatible']


def test_fragments_are_position_independent_but_ordered():
    prototypes = {0:'AAWGQQC', 1:'CWGAA', 2:'WGCC', 3:'WGAAAAC'}
    assert fragment_compatible_classes(('WG','C'),prototypes) == {0,2,3}

@pytest.mark.parametrize('fragments', [(), ('',), ('WX',)])
def test_no_evidence_is_not_an_all_class_label(fragments):
    with pytest.raises(TrainingDataError): fragment_compatible_classes(fragments,{0:'AA'})


def test_ambiguous_reference_residues_are_skipped(crop_case):
    ref, record, _ = crop_case()
    germline = ref.germlines.genes['J']['J1']
    ref.germlines.genes['J']['J1'] = replace(germline,aa_seq='WXQAA')
    evidence = reference_residue_evidence(record,'J',ref,10,13)
    assert evidence.fragments == ('W','Q') and evidence.positions == (0,2)


def test_partial_d_is_masked_and_full_d_retains_allele_target(crop_case):
    ref, record, builder = crop_case()
    base = {**builder(record).sample, 'd_start':7,'d_end':9,'d_class':2,'d_valid':1.}
    partial = crop_protein_sample(record,base,ref,8,14)
    assert partial['d_valid'] == 0 and not partial['d_class_supervise']
    full = crop_protein_sample(record,base,ref,7,14)
    assert full['d_valid'] == 1 and full['d_compatible'] == {2}


def test_rehearsal_and_crop_are_reproducible_and_do_not_change_input(crop_case):
    ref, record, builder = crop_case()
    original = deepcopy(record)
    for p_crop in (0.,1.):
        policy=ProteinCropPolicy(p_crop,1.,1.)
        a = ProteinCropAugmenter(ref,builder,np.random.default_rng(21),policy)(record).sample
        b = ProteinCropAugmenter(ref,builder,np.random.default_rng(21),policy)(record).sample
        np.testing.assert_array_equal(a.pop('tokens'),b.pop('tokens'))
        assert a == b
        if not p_crop:
            assert a['j_compatible'] == {ref.class_maps['J'].class_of('J1')}
            assert a['j_class_reason'] == 'source_label'
    assert record == original


def test_rejections_pass_through_the_shared_builder(crop_case):
    ref, record, builder = crop_case()
    record['productive']=False
    assert ProteinCropAugmenter(ref,builder,np.random.default_rng(0))(record).reason == 'not_productive'

@pytest.mark.parametrize('probability',[-.1,1.1,float('nan'),float('inf'),True,None,'0.5'])
def test_bad_augmentation_probabilities_are_configuration_errors(probability):
    with pytest.raises(ConfigurationError): ProteinCropPolicy(crop_probability=probability)

@pytest.mark.parametrize('lo,hi', [(-1,3),(0,16),(5,5),(True,5),(.5,4)])
def test_invalid_evidence_window_is_rejected(crop_case,lo,hi):
    ref,record,_=crop_case()
    with pytest.raises(TrainingDataError): reference_residue_evidence(record,'J',ref,lo,hi)


def test_source_class_must_remain_in_compatible_targets(crop_case):
    ref,record,builder=crop_case()
    base=builder(record).sample
    base['j_class']=ref.class_maps['J'].class_of('J2')
    with pytest.raises(TrainingDataError,match='exclude their source'):
        crop_protein_sample(record,base,ref,10,13)


def test_class_only_mask_survives_batching_with_rehearsal_defaults(crop_case):
    from alignair.data.adapters import CategoricalBatchAdapter
    ref,record,builder=crop_case()
    base=builder(record).sample
    record['j_cigar']=None
    crop=crop_protein_sample(record,base,ref,10,13)
    # Mixed batches can contain ordinary singleton samples without an explicit class-only mask.
    rehearsal={**base,'v_compatible':{base['v_class']},'j_compatible':{base['j_class']}}
    batch=CategoricalBatchAdapter(64,class_counts={'v':1,'j':3})([crop,rehearsal])
    assert batch.targets['j_class_supervise'].tolist()==[False,True]
    assert batch.targets['j_valid'].tolist()==[1.,1.]

@pytest.mark.parametrize('value',[.5,float('nan'),'false',None])
def test_batch_adapter_rejects_nonbinary_class_masks(crop_case,value):
    from alignair.data.adapters import CategoricalBatchAdapter
    ref,record,builder=crop_case()
    base={**builder(record).sample,'j_class_supervise':value}
    with pytest.raises(TrainingDataError,match='binary'):
        CategoricalBatchAdapter(64)([base])


def test_evidence_for_absent_gene_and_unknown_allele(crop_case):
    ref,record,_=crop_case()
    record.update(j_sequence_start=None,j_sequence_end=None)
    assert reference_residue_evidence(record,'J',ref,10,13).reason=='gene_absent'
    record.update(j_sequence_start=30,j_sequence_end=45,j_call='missing')
    with pytest.raises(TrainingDataError,match='absent from the protein reference'):
        reference_residue_evidence(record,'J',ref,10,13)

@pytest.mark.parametrize('lo,hi',[(-1,4),(0,16),(0,0),(False,3)])
def test_invalid_crop_window_raises_typed_error(crop_case,lo,hi):
    ref,record,builder=crop_case()
    with pytest.raises(TrainingDataError):crop_protein_sample(record,builder(record).sample,ref,lo,hi)


def test_alignment_contract_errors_include_record_and_gene(crop_case):
    ref,record,_=crop_case()
    record.update(sequence_id='sample-7',j_cigar='garbage')
    with pytest.raises(TrainingDataError) as exc:
        reference_residue_evidence(record,'J',ref,10,13)
    assert exc.value.context['record_id']=='sample-7' and exc.value.context['gene']=='J'


def test_augmenter_rejects_policy_and_reference_mismatches(crop_case):
    ref,record,builder=crop_case()
    with pytest.raises(ConfigurationError,match='ProteinCropPolicy'):
        ProteinCropAugmenter(ref,builder,np.random.default_rng(0),None)
    maps=dict(builder.class_maps)
    maps['J']=replace(maps['J'],db_hash='different')
    with pytest.raises(ConfigurationError,match='label spaces disagree'):
        ProteinCropAugmenter(ref,replace(builder,class_maps=maps),np.random.default_rng(0))
    with pytest.raises(ConfigurationError,match='D vocabularies disagree'):
        ProteinCropAugmenter(ref,replace(builder,d_index={'D1':0}),np.random.default_rng(0))


def test_tensor_tokenizer_preserves_crop_and_reports_length_changing_corruption(crop_case):
    import torch
    ref,record,builder=crop_case()
    original=builder.tokenizer
    class TensorTokenizer:
        def encode(self,sequence):return torch.tensor(original.encode(sequence))
    builder=replace(builder,tokenizer=TensorTokenizer())
    sample=ProteinCropAugmenter(ref,builder,np.random.default_rng(0),ProteinCropPolicy(1.,0.,0.))(record).sample
    assert isinstance(sample['tokens'],torch.Tensor) and len(sample['tokens']) < 15
    class ChangingTokenizer:
        def __init__(self):self.calls=0
        def encode(self,sequence):
            self.calls+=1
            return original.encode(sequence if self.calls==1 else sequence+'A')
    builder=replace(builder,tokenizer=ChangingTokenizer())
    with pytest.raises(TrainingDataError,match='preserve residue positions'):
        ProteinCropAugmenter(ref,builder,np.random.default_rng(0),ProteinCropPolicy(1.,0.,0.))(record)


def test_trimmed_v_reads_are_eligible_for_cropping(crop_case):
    ref,record,builder=crop_case()
    record.update(v_germline_start=6,v_germline_end=30,v_sequence_end=24,v_cigar='24M')
    sample=ProteinCropAugmenter(ref,builder,np.random.default_rng(0),ProteinCropPolicy(1.,0.,0.))(record).sample
    assert len(sample['tokens']) < len(record['sequence_aa'])
