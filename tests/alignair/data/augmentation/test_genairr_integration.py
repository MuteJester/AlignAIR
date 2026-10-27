"""Native CIGAR/frame contracts must yield usable or explicitly masked class targets."""
from collections import Counter
from itertools import islice
import numpy as np
import pytest
from alignair.reference import build_protein_reference
from alignair.data.sampling import ProteinSampleBuilder, ProteinSamplePolicy
from alignair.data.tokenizer import AAProteinTokenizer
from alignair.data.augmentation import ProteinCropAugmenter, ProteinCropPolicy
from alignair.aa.train_gym import _usable_stream

pytestmark=pytest.mark.integration


def test_native_cropped_reads_never_exclude_their_source_class():
    reference=build_protein_reference('HUMAN_IGH_OGRDB')
    builder=ProteinSampleBuilder(reference.class_maps,AAProteinTokenizer(),192,
        policy=ProteinSamplePolicy(require_productive=False,allow_stops=True))
    augmenter=ProteinCropAugmenter(reference,builder,np.random.default_rng(17),ProteinCropPolicy(1.,0.,0.))
    outcomes=Counter()
    for record in islice(_usable_stream('HUMAN_IGH_OGRDB',(.3,.6,.9),.25,17,1,None),24):
        result=augmenter(record)
        if result.sample is None:
            continue
        sample=result.sample
        for gene in ('v','j'):
            outcomes[sample[f'{gene}_class_reason']]+=1
            if sample[f'{gene}_class_supervise']:
                expected=reference.class_maps[gene.upper()].class_of(record[f'{gene}_call'].split(',')[0])
                assert expected in sample[f'{gene}_compatible']
            else:
                assert sample[f'{gene}_class']==-1 and not sample[f'{gene}_compatible']
    assert outcomes['reference_fragments'] > 0
    assert outcomes['no_reference_codon_evidence'] > 0
