import time
import torch
from alignair.nn.aligner.pointer import BandedPointerAligner
from alignair.nn.aligner.soft_dp import SoftDPAligner


def test_pointer_forward_is_much_faster_than_softdp_on_cpu():
    B, S, Lg, d = 8, 64, 80, 32
    seg = torch.randn(B, S, d); germ = torch.randn(B, Lg, d)
    sm = torch.ones(B, S, dtype=torch.bool); gm = torch.ones(B, Lg, dtype=torch.bool)
    ptr = BandedPointerAligner(d_model=d); sdp = SoftDPAligner(d_model=d)

    def _time(fn, n=3):
        fn()                                  # warmup
        t = time.perf_counter()
        for _ in range(n):
            fn()
        return (time.perf_counter() - t) / n

    tp = _time(lambda: ptr(seg, sm, germ, gm))
    ts = _time(lambda: sdp(seg, sm, germ, gm))
    assert tp < ts / 5, f"pointer {tp*1e3:.1f}ms not <<= softdp {ts*1e3:.1f}ms"
