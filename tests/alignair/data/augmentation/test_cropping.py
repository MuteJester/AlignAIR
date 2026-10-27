import pytest
from alignair.data.augmentation.cropping import compatible_classes, crop_window, shift_clip

pytestmark = pytest.mark.component


def test_crop_window_kinds():
    assert crop_window(100, "nterm", 40) == (0, 40)          # keep prefix
    assert crop_window(100, "cterm", 40) == (60, 100)        # keep suffix
    lo, hi = crop_window(100, "internal", 40, start=30)
    assert (lo, hi) == (30, 70)


def test_shift_clip_maps_and_drops_absent():
    # V=[0,50) J=[60,90) in a 100-residue read; keep the prefix [0,55): V fully in, J absent.
    assert shift_clip((0, 50), 0, 55) == (0, 50)
    assert shift_clip((60, 90), 0, 55) is None               # J entirely outside the kept window
    # keep the suffix [45,100): V partially in (residues 45..50 -> shifted 0..5), J fully in
    assert shift_clip((0, 50), 45, 100) == (0, 5)
    assert shift_clip((60, 90), 45, 100) == (15, 45)


def test_compatible_classes_grows_as_the_window_shrinks():
    # prototypes over a shared prefix: classes 0,1 identical on [0,3), diverge at residue 3
    protos = {0: "MKVCAA", 1: "MKVCLL", 2: "EQLDXX"}
    # full-length window distinguishes 0 from 1
    assert compatible_classes(0, 0, 6, protos) == {0}
    # observing only [0,4) still distinguishes (they differ at residue 4)... they share [0,4)="MKVC"
    assert compatible_classes(0, 0, 4, protos) == {0, 1}     # 0 and 1 share the prefix -> ambiguous
    # a coverage window that includes the divergent residue 4 separates them
    assert compatible_classes(0, 0, 5, protos) == {0}
    # class 2 is never compatible with class 0
    assert 2 not in compatible_classes(0, 0, 3, protos)


@pytest.mark.parametrize('n,kind,keep,start',[(0,'nterm',1,None),(True,'cterm',3,None),
    (5,'other',3,None),(5,'internal',1.5,None),(5,'internal',2,False)])
def test_invalid_crop_configuration(n,kind,keep,start):
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError): crop_window(n,kind,keep,start)


def test_internal_default_and_clamping():
    assert crop_window(9,'internal',3) == (3,6)
    assert crop_window(9,'internal',99,start=-5) == (0,9)
    assert crop_window(9,'internal',0,start=99) == (8,9)

@pytest.mark.parametrize('span,lo,hi',[((-1,3),0,4),((0,3),2,2),((0,),0,4),((0,3),False,4)])
def test_invalid_clipped_coordinates_raise(span,lo,hi):
    from alignair.errors import TrainingDataError
    with pytest.raises(TrainingDataError): shift_clip(span,lo,hi)

@pytest.mark.parametrize('lo,hi',[(2,2),(0,99),(-1,3),(0,False)])
def test_legacy_reference_window_cannot_silently_slice_to_empty(lo,hi):
    from alignair.errors import TrainingDataError
    with pytest.raises(TrainingDataError): compatible_classes(0,lo,hi,{0:'ACD'})


def test_legacy_reference_window_requires_true_class():
    from alignair.errors import ConfigurationError
    with pytest.raises(ConfigurationError): compatible_classes(4,0,2,{0:'ACD'})


def test_historical_modules_only_reexport_the_canonical_operators():
    from alignair.aa import augment, truncation
    from alignair.data.augmentation import protein, cropping
    for name in augment.__all__: assert getattr(augment,name) is getattr(protein,name)
    for name in truncation.__all__: assert getattr(truncation,name) is getattr(cropping,name)
