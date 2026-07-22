import numpy as np, torch
from alignair.aa.resume import save_checkpoint, load_checkpoint

def _tiny():
    m = torch.nn.Linear(4, 3)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=100, eta_min=1e-5)
    # take one step so optimizer has non-empty moment state
    loss = m(torch.randn(2, 4)).sum(); loss.backward(); opt.step(); sch.step()
    return m, opt, sch

def test_full_state_roundtrip(tmp_path):
    m, opt, sch = _tiny()
    p = str(tmp_path / "ck.pt")
    save_checkpoint(p, model=m, cfg={"max_seq_length": 192}, step=5, optimizer=opt,
                    scheduler=sch, lr=1e-4, history=[{"step": 5, "total": 1.0}],
                    provenance={"seed": 0})
    ck = load_checkpoint(p)
    assert ck["step"] == 5 and ck["provenance"]["seed"] == 0
    # optimizer moment state present and restorable
    m2 = torch.nn.Linear(4, 3); opt2 = torch.optim.AdamW(m2.parameters(), lr=1e-4)
    opt2.load_state_dict(ck["optimizer"])
    assert opt2.state_dict()["state"], "optimizer moment state must be preserved"
    assert ck["scheduler"]["T_max"] == 100

def test_old_format_load_is_graceful(tmp_path):
    p = str(tmp_path / "old.pt")
    torch.save({"model": torch.nn.Linear(4, 3).state_dict(), "cfg": {}, "step": 40000}, p)
    ck = load_checkpoint(p)
    assert ck["step"] == 40000 and ck["optimizer"] is None and ck["scheduler"] is None
