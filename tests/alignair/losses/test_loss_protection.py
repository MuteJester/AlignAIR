import math
from alignair.losses.dnalignair_loss import DNAlignAIRLoss


def test_protected_heads_have_tighter_cap_and_higher_weight_floor():
    loss = DNAlignAIRLoss(has_d=True, protected_max_log_var=1.5)
    assert "v_match" in loss.protected_heads
    assert "v_germline" in loss.protected_heads and "d_germline" in loss.protected_heads
    assert loss.weights["v_match"].max_log_var == 1.5         # protected -> tighter
    assert loss.weights["region"].max_log_var == 3.0          # unprotected -> default
    # tighter cap => strictly higher minimum precision weight (can't be abandoned)
    assert math.exp(-1.5) > math.exp(-3.0)


def test_freeze_toggles_all_heads():
    loss = DNAlignAIRLoss(has_d=False)
    loss.set_log_vars_frozen(True)
    assert all(w._frozen for w in loss.weights.values())
    loss.set_log_vars_frozen(False)
    assert not any(w._frozen for w in loss.weights.values())
