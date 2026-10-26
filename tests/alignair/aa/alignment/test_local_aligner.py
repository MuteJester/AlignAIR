import pytest
from alignair.aa.local_aligner import classify_v, v_class_prototypes
from alignair.aa.reference import AAGermline

pytestmark = pytest.mark.component


def _germlines():
    # two synonymous alleles (class 0, identical protein) + one distinct (class 1)
    return ({"a*01": AAGermline("a*01", "QVQLVESGGGLVQP", 0, 13, "V"),
             "a*02": AAGermline("a*02", "QVQLVESGGGLVQP", 0, 13, "V"),
             "b*01": AAGermline("b*01", "EVQLLESGGGLIKP", 0, 13, "V")},
            {"a*01": 0, "a*02": 0, "b*01": 1})


def test_prototypes_are_one_per_class():
    germ, cof = _germlines()
    classes, protos = v_class_prototypes(germ, cof)
    assert classes == [0, 1]
    assert protos[0] == "QVQLVESGGGLVQP" and protos[1] == "EVQLLESGGGLIKP"


def test_exact_protein_selects_its_class():
    germ, cof = _germlines()
    classes, protos = v_class_prototypes(germ, cof)
    c, _ = classify_v("QVQLVESGGGLVQP", classes, protos)
    assert c == 0


def test_local_alignment_tolerates_an_indel_and_still_calls_the_class():
    germ, cof = _germlines()
    classes, protos = v_class_prototypes(germ, cof)
    # a single-residue insertion in the class-0 protein: local alignment gaps it out
    c, _ = classify_v("QVQLVESGGXGLVQP", classes, protos)
    assert c == 0
    # the class-1 protein still routes to class 1
    c1, _ = classify_v("EVQLLESGGGLIKP", classes, protos)
    assert c1 == 1


def test_stop_and_unknown_chars_do_not_crash():
    germ, cof = _germlines()
    classes, protos = v_class_prototypes(germ, cof)
    c, _ = classify_v("QVQLVE*GGGLVQPZZ", classes, protos)   # '*', unknown 'Z'
    assert c in (0, 1)
