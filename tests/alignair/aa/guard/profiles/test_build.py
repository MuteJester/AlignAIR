import pytest
from alignair.aa.guard.profiles import build_bank_fixture, build_profile

pytestmark = pytest.mark.component

# aligned (equal-length) MSA fixtures — burned-dev-style toy V and J segments
MSA_V = (b">a\nQVQLVQSGAEVKKPGASVKVSCKASGYTF\n"
         b">b\nQVQLVQSGAEVKKPGSSVKVSCKASGGTF\n"
         b">c\nEVQLVQSGAEVKKPGASVKVSCKASGYTL\n")
MSA_J = (b">j1\nWGQGTLVTVSSASTKG\n"
         b">j2\nWGQGTTVTVSSASTKG\n"
         b">j3\nWGQGTMVTVSSASTKG\n")


def test_build_is_deterministic():
    _, h1 = build_profile("human_igh", "v", MSA_V)
    _, h2 = build_profile("human_igh", "v", MSA_V)
    assert h1 == h2 and len(h1) == 64


def test_v_only_bank_cannot_claim_complete_domain():
    bank = build_bank_fixture(regions={"human_igh": {"v": MSA_V}})
    assert not bank.supports_complete_domain("human_igh")


def test_paired_vj_bank_supports_complete_domain():
    bank = build_bank_fixture(regions={"human_igh": {"v": MSA_V, "j": MSA_J}})
    assert bank.supports_complete_domain("human_igh")


def test_igsf_full_domain_supports_complete_domain():
    bank = build_bank_fixture(regions={"igsf_vset": {"domain": MSA_V}})
    assert bank.supports_complete_domain("igsf_vset")
    # but a receptor group needs a J even if it has a full-length V
    v_only = build_bank_fixture(regions={"human_igh": {"v": MSA_V}})
    assert not v_only.supports_complete_domain("human_igh")


def test_bank_checksum_is_deterministic_and_region_sensitive():
    b1 = build_bank_fixture(regions={"human_igh": {"v": MSA_V, "j": MSA_J}})
    b2 = build_bank_fixture(regions={"human_igh": {"v": MSA_V, "j": MSA_J}})
    assert b1.checksum() == b2.checksum() and len(b1.checksum()) == 64
    # dropping J changes the bank checksum
    b3 = build_bank_fixture(regions={"human_igh": {"v": MSA_V}})
    assert b3.checksum() != b1.checksum()
