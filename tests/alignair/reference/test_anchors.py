"""``annotate`` fills each V and J allele's missing anchor (and a V's missing IMGT-gapped form) one way, for training
and for inference: a supplied value is kept; else the known allele with the same sequence gives its own; else
GenAIRR's resolver derives it (V from its IMGT gaps, J from a unique F/W-G-X-G motif); else an ungapped V takes the
gaps of its nearest same-length known allele. Each allele records how its anchor was found; ``anchor_report`` counts
the anchored alleles and names those without."""
import GenAIRR.data as gd
import pytest

from alignair.reference import AnchorEvidence
from alignair.reference.anchors import annotate, anchor_report
from alignair.reference.reference_set import GeneReference, ReferenceSet

pytestmark = pytest.mark.component


@pytest.fixture(scope="module")
def known():
    return ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB)


def _toy_known():
    v = GeneReference(["IGHV1-1*01", "IGHV2-1*01"], ["ACGTACGTAC", "TTTTGGGGCC"], {"IGHV1-1*01": 0, "IGHV2-1*01": 1},
                      anchors={"IGHV1-1*01": 4, "IGHV2-1*01": 5},
                      gapped={"IGHV1-1*01": "ACGT..ACGTAC", "IGHV2-1*01": "TTTTG..GGGCC"})
    d = GeneReference(["IGHD1-1*01"], ["GGGCCC"], {"IGHD1-1*01": 0})
    j = GeneReference(["IGHJ1*01"], ["TTTGGAA"], {"IGHJ1*01": 0}, anchors={"IGHJ1*01": 3})
    return ReferenceSet({"V": v, "D": d, "J": j}, has_d=True)


def _users(v, j=None, d=True):
    return ReferenceSet.from_alleles({"V": v, **({"D": {"IGHD1-1*01": "GGGCCC"}} if d else {}),
                                       "J": j or {"IGHJ1*01": "TTTGGAA"}})


def _evidence(reference, gene, name):
    return reference.gene(gene).info[name].anchor_evidence


def _snp(sequence, at=100):
    return sequence[:at] + ("A" if sequence[at] != "A" else "C") + sequence[at + 1:]


# --- the cases complete_reference covered, now annotate's ---------------------------------------------------------

def test_a_known_allele_keeps_its_gapped_form_and_anchor():
    done = annotate(_users({"IGHV1-1*01": "ACGTACGTAC"}), _toy_known())
    assert done.gene("V").gapped == {"IGHV1-1*01": "ACGT..ACGTAC"}
    assert done.gene("V").anchors == {"IGHV1-1*01": 4} and done.gene("J").anchors == {"IGHJ1*01": 3}


def test_a_snp_variant_takes_its_nearest_alleles_gaps_with_its_own_bases():
    done = annotate(_users({"IGHV1-1*novel": "ACGAACGTAC"}), _toy_known())          # one SNP from IGHV1-1*01
    assert done.gene("V").gapped == {"IGHV1-1*novel": "ACGA..ACGTAC"}
    assert done.gene("V").anchors == {"IGHV1-1*novel": 4}


def test_a_variant_differing_at_more_than_a_tenth_of_its_bases_takes_nothing():
    done = annotate(_users({"IGHV1-1*far": "ACGAACGTAA"}), _toy_known())            # two SNPs in ten bases
    assert not done.gene("V").gapped and not done.gene("V").anchors


def test_a_gapped_sequence_keeps_its_gaps():
    done = annotate(_users({"IGHV9-9*01": "ACGT..ACGTCC"}), _toy_known())
    assert done.gene("V").sequences == ["ACGTACGTCC"] and done.gene("V").gapped == {"IGHV9-9*01": "ACGT..ACGTCC"}


def test_an_allele_without_a_same_length_neighbour_stays_without_gaps_or_anchor():
    done = annotate(_users({"IGHV7-7*01": "ACGTACG"}), _toy_known())
    assert done.gene("V").sequences == ["ACGTACG"]
    assert not done.gene("V").gapped and not done.gene("V").anchors


# --- the order and the evidence -----------------------------------------------------------------------------------

def test_a_supplied_anchor_and_gapped_form_are_never_overwritten(known):
    v = known.gene("V")
    user = ReferenceSet.from_alleles({"V": {v.names[0]: v.sequences[0]}, "J": {}}, anchors={"V": {v.names[0]: 5}})
    done = annotate(user, known)
    assert done.gene("V").anchors[v.names[0]] == 5
    assert _evidence(done, "V", v.names[0]) == AnchorEvidence("supplied", "supplied")


def test_an_allele_the_known_reference_holds_takes_its_gapped_form_and_anchor(known):
    v = known.gene("V")
    v0, s = v.names[0], v.sequences[0]
    for name in (v0, "IGHV1-NOVEL*01"):
        done = annotate(ReferenceSet.from_alleles({"V": {name: s}, "J": {}}), known)
        assert done.gene("V").gapped[name] == v.gapped[v0] and done.gene("V").anchors[name] == 288
        assert _evidence(done, "V", name) == AnchorEvidence("known_allele", "inferred", None, "IGHVF1-G1*01")


def test_the_known_allele_wins_over_a_disagreeing_motif():
    cow = ReferenceSet.from_dataconfigs(gd.COW_TCRA_IMGT)
    j = cow.gene("J")
    user = ReferenceSet.from_alleles({"V": {}, "J": {"TRAJ10*01": j.sequences[j.index["TRAJ10*01"]]}})
    assert annotate(user, cow).gene("J").anchors["TRAJ10*01"] == 5
    assert annotate(user).gene("J").anchors["TRAJ10*01"] == 29


def test_a_gapped_v_is_anchored_from_its_imgt_gaps(known):
    v = known.gene("V")
    gapped = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": _snp(v.sequences[0])}, "J": {}}),
                      known).gene("V").gapped["IGHV1-NOVEL*01"]
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": gapped}, "J": {}}))
    assert done.gene("V").anchors["IGHV1-NOVEL*01"] == 288
    assert _evidence(done, "V", "IGHV1-NOVEL*01").method == "imgt_gapped_104"


def test_a_j_is_anchored_from_its_unique_motif(known):
    j = known.gene("J")
    done = annotate(ReferenceSet.from_alleles({"V": {}, "J": {"IGHJ1*91": j.sequences[0]}}))
    assert done.gene("J").anchors["IGHJ1*91"] == 18
    assert _evidence(done, "J", "IGHJ1*91").method == "unique_j_fwgxg"


def test_a_j_with_two_motifs_is_ambiguous_and_unanchored(known):
    js = known.gene("J").sequences[0]
    done = annotate(ReferenceSet.from_alleles({"V": {}, "J": {"IGHJ1*91": js[:18] + "TGGGGCAAAGGC" + js[18:]}}),
                    known)
    assert not done.gene("J").anchors
    assert _evidence(done, "J", "IGHJ1*91").status == "ambiguous"


def test_a_v_without_its_cys_is_anchored_at_its_imgt_position_gapped_or_not(known):
    """Codon 104 anchors the V whatever its residue (IMGT numbers the junction from it), where GenAIRR calls such a
    gapped V invalid; given gapped or ungapped, the allele gets the same anchor, and the report names it as
    non-canonical."""
    v = known.gene("V")
    g = v.gapped[v.names[0]]
    gapped = g[:309] + "TGG" + g[312:]
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": gapped}, "J": {}}), known)
    assert done.gene("V").anchors["IGHV1-NOVEL*01"] == 288
    evidence = _evidence(done, "V", "IGHV1-NOVEL*01")
    assert (evidence.method, evidence.status) == ("imgt_position_104", "inferred") and "TGG" in evidence.detail
    ungapped = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": gapped.replace(".", "")}, "J": {}}), known)
    assert ungapped.gene("V").anchors["IGHV1-NOVEL*01"] == 288
    report = anchor_report(done)
    assert report.complete() and report.noncanonical == {"V": ("IGHV1-NOVEL*01",)}
    assert "codon 104 not Cys: IGHV1-NOVEL*01" in report.summary()


def test_a_v_gapped_off_imgt_numbering_stays_unanchored(known):
    """A gapped form whose Cys sits a codon or more past column 309 is not on IMGT numbering (GenAIRR G2): no anchor
    at column 309 for it."""
    v = known.gene("V")
    g = v.gapped[v.names[0]]
    shifted = g[:300] + "..." + g[300:-3]
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": shifted}, "J": {}}))
    assert not done.gene("V").anchors
    assert _evidence(done, "V", "IGHV1-NOVEL*01").status == "invalid"


def test_an_ungapped_snp_variant_takes_its_nearest_alleles_gaps_and_anchor(known):
    snp = _snp(known.gene("V").sequences[0])
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": snp}, "J": {}}), known)
    gapped = done.gene("V").gapped["IGHV1-NOVEL*01"]
    assert len(gapped) == 322 and gapped.count(".") == 21 and gapped.replace(".", "") == snp
    assert done.gene("V").anchors["IGHV1-NOVEL*01"] == 288
    assert _evidence(done, "V", "IGHV1-NOVEL*01") == AnchorEvidence(
        "nearest_known_allele", "inferred", "1 of 301 bases differ", "IGHVF1-G1*01")


def test_an_indel_variant_stays_unanchored_until_genairr_gaps_it(known):
    """GenAIRR's G6 (gap an ungapped V by its nearest gapped template) will give this allele its gapped form and
    anchor 285; until then it is reported unresolved. When G6 lands, expect them here."""
    s = known.gene("V").sequences[0]
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": s[:250] + s[253:]}, "J": {}}), known)
    assert not done.gene("V").gapped and not done.gene("V").anchors
    evidence = _evidence(done, "V", "IGHV1-NOVEL*01")
    assert (evidence.method, evidence.status) == ("unresolved", "unresolved")


def test_the_report_counts_anchors_and_names_the_alleles_without(known):
    v, j = known.gene("V"), known.gene("J")
    s, js = v.sequences[0], j.sequences[0]
    user = ReferenceSet.from_alleles({"V": {v.names[0]: s, "IGHVF1-G1*91": _snp(s), "IGHVF1-G1*92": s[:250] + s[253:]},
                                       "J": {j.names[0]: js, "IGHJ1*91": js[:18] + "TGGGGCAAAGGC" + js[18:]}})
    report = anchor_report(annotate(user, known))
    assert report.anchored == {"V": 2, "J": 1} and report.total == {"V": 3, "J": 2}
    assert report.missing == {"V": {"unresolved": ("IGHVF1-G1*92",)}, "J": {"ambiguous": ("IGHJ1*91",)}}
    assert not report.complete()
    assert report.summary() == "V 2/3, J 1/2 anchored; no anchor: IGHVF1-G1*92 (unresolved), IGHJ1*91 (ambiguous)"


def test_annotate_keeps_order_loci_provenance_and_d_and_leaves_its_input_untouched(known):
    v = known.gene("V")
    user = ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": _snp(v.sequences[0]), v.names[1]: v.sequences[1]},
                                       "D": {"IGHD1-1*01": "GGGCCC"}, "J": {}})
    before = user.to_serializable()
    done = annotate(user, known)
    assert user.to_serializable() == before
    assert done.gene("V").names == user.gene("V").names
    assert [(l.locus, l.ranges) for l in done.loci] == [(l.locus, l.ranges) for l in user.loci]
    assert done.loci[0].species == known.loci[0].species          # a model's schema is the bound reference's
    assert done.provenance == user.provenance and done.gene("D").names == ["IGHD1-1*01"]
    assert not done.gene("D").anchors and not done.gene("D").info
    assert annotate(done, known).to_serializable() == done.to_serializable()


def test_the_summary_names_five_alleles_then_how_many_more(known):
    s = known.gene("V").sequences[0]
    user = ReferenceSet.from_alleles({"V": {f"IGHVF1-G1*9{i}": s[:250] + s[253 + i:] for i in range(7)}, "J": {}})
    summary = anchor_report(annotate(user, known)).summary()
    assert summary.startswith("V 0/7, J 0/0 anchored; no anchor: IGHVF1-G1*90 (unresolved), ")
    assert summary.endswith("IGHVF1-G1*94 (unresolved), and 2 more")
    complete = anchor_report(annotate(ReferenceSet.from_alleles({"V": {"IGHVF1-G1*01": s}, "J": {}}), known))
    assert complete.complete() and complete.summary() == "V 1/1, J 0/0 anchored"


# --- the final review's fix pass ----------------------------------------------------------------------------------

def test_a_models_own_trd_alleles_as_a_fasta_keep_the_models_locus_schema(tmp_path):
    """TRD's V alleles include TRAV/DV ones, whose names show TRA: bound to a TRD model they stay in its one locus
    (the chain head's class), not a TRA locus the model has none of."""
    from alignair.model_file import serialize
    trd = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRD_IMGT)
    path = tmp_path / "trd.fasta"
    path.write_text(serialize.reference_fasta(trd))
    bound = annotate(ReferenceSet.from_fasta(str(path)), trd)
    assert [(l.locus, l.ranges) for l in bound.loci] == [(l.locus, l.ranges) for l in trd.loci]
    assert all(bound.gene(g).names == trd.gene(g).names for g in ("V", "D", "J"))


def test_a_reference_bound_to_a_two_locus_model_takes_its_loci_in_its_order():
    """The model's loci, in its order, stay the bound reference's (the chain head's classes), even one the reference
    has no allele of; an allele the model holds takes its locus, a novel one its name's."""
    igk_igl = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    v, j = igk_igl.gene("V"), igk_igl.gene("J")
    user = ReferenceSet.from_alleles({"V": {"IGLV9-99*01": _snp(v.sequences[200], 50), v.names[0]: v.sequences[0]},
                                       "J": {j.names[0]: j.sequences[0]}})
    bound = annotate(user, igk_igl)
    assert bound.locus_names() == ("IGK", "IGL")
    assert bound.gene("V").names == [v.names[0], "IGLV9-99*01"]
    assert [l.ranges for l in bound.loci] == [{"V": (0, 1), "J": (0, 1)}, {"V": (1, 2)}]


def test_an_allele_no_locus_of_a_two_locus_model_holds_is_refused():
    """A one-locus model takes a novel allele into its locus whatever its name shows (a TRD model's TRAV/DV alleles
    are named TRA); a model of several loci refuses one whose name shows another locus (the locus it does not call)
    or none (the allele, which no locus of the model holds)."""
    from alignair.errors import ConfigurationError, InputError
    igk_igl = ReferenceSet.from_dataconfigs(gd.HUMAN_IGK_OGRDB, gd.HUMAN_IGL_OGRDB)
    user = ReferenceSet.from_alleles({"V": {"TRBV9-99*01": "ACGT" * 70}, "J": {}})
    with pytest.raises(ConfigurationError, match=r"alleles of TRB, a locus the model does not call"):
        annotate(user, igk_igl)
    unnamed = ReferenceSet.from_alleles({"V": {"MYV9*01": "ACGT" * 70}, "J": {}})
    with pytest.raises(InputError, match=r"MYV9\*01"):
        annotate(unnamed, igk_igl)
    trd = ReferenceSet.from_dataconfigs(gd.HUMAN_TCRD_IMGT)
    novel = ReferenceSet.from_alleles({"V": {"TRAV14/DV4*99": "ACGT" * 70}, "J": {}})
    assert annotate(novel, trd).locus_names() == ("TRD",)


def test_a_known_allele_whose_gapped_form_is_not_its_sequence_lends_no_gaps():
    """HUMAN_IGH_EXTENDED's IGHV4-3GMG*02/03/04 gapped forms hold more bases than their sequences: such a form cannot
    carry another allele's bases, and binding a variant of one does not crash."""
    union = ReferenceSet.from_dataconfigs(gd.HUMAN_IGH_OGRDB, gd.HUMAN_IGH_EXTENDED)
    v = union.gene("V")
    name = "IGHV4-3GMG*04"
    assert v.gapped[name].replace(".", "") != v.sequences[v.index[name]]
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV4-3GMG*99": _snp(v.sequences[v.index[name]])}, "J": {}}),
                    union)
    gapped = (done.gene("V").gapped or {}).get("IGHV4-3GMG*99")
    assert gapped is None or gapped.replace(".", "") == _snp(v.sequences[v.index[name]])


def test_gaps_carried_without_an_anchor_name_the_allele_that_lent_them():
    known = _toy_known()
    known.gene("V").anchors.pop("IGHV1-1*01")
    done = annotate(_users({"IGHV1-1*novel": "ACGAACGTAC"}), known)
    assert done.gene("V").gapped == {"IGHV1-1*novel": "ACGA..ACGTAC"} and not done.gene("V").anchors
    assert _evidence(done, "V", "IGHV1-1*novel") == AnchorEvidence(
        "nearest_known_allele", "unresolved", "1 of 10 bases differ", "IGHV1-1*01")


def test_a_v_that_takes_its_nearest_alleles_anchor_asks_genairr_nothing(known, monkeypatch):
    """An ungapped variant gapped by its nearest known allele needs no resolver call (GenAIRR anchors a V from its
    gapped form only)."""
    from alignair.genairr import anchors as adapter
    calls = []
    resolve = adapter.resolve_anchor
    monkeypatch.setattr(adapter, "resolve_anchor", lambda *a, **k: calls.append(a) or resolve(*a, **k))
    done = annotate(ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": _snp(known.gene("V").sequences[0])}, "J": {}}),
                    known)
    assert done.gene("V").anchors["IGHV1-NOVEL*01"] == 288 and calls == []


def test_annotate_shares_no_gene_or_locus_object_with_its_input(known):
    user = ReferenceSet.from_alleles({"V": {"IGHV1-NOVEL*01": _snp(known.gene("V").sequences[0])},
                                       "D": {"IGHD1-1*01": "GGGCCC"}, "J": {}})
    done = annotate(user)
    assert done.gene("D") is not user.gene("D") and all(a is not b for a, b in zip(done.loci, user.loci))
