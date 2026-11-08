"""cibuildwheel smoke test: verify the compiled kernels (germline CIGAR, the V re-score's band) loaded in the built
wheel, WITHOUT importing the full ``alignair`` package (which pulls heavy runtime deps like torch)."""
import glob
import importlib.util
import os
import sys

import numpy as np

origin = importlib.util.find_spec("alignair").origin          # locate, don't execute (no torch import)


def load(package, name):
    matches = glob.glob(os.path.join(os.path.dirname(origin), package, f"{name}*.*"))
    so = [m for m in matches if m.endswith((".so", ".pyd"))]
    if not so:
        sys.exit(f"compiled {name} kernel missing from the wheel (found: {matches})")
    spec = importlib.util.spec_from_file_location(name, so[0])   # name must match PyInit_<name>
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    print("cython kernel OK:", os.path.basename(so[0]))
    return mod


derive = load("predict", "_derive_cy")
assert derive.derive_cigar(b"ACGGT", b"ACGT", 1) == "2M1I2M", derive.derive_cigar(b"ACGGT", b"ACGT", 1)

band = load("refine", "_vband_cy")
read = np.array([[1, 3, 4, 2]], dtype=np.int64)                # the read is its row: four matches on one diagonal
one, zero = np.array([0], dtype=np.int64), np.zeros((1, 1), dtype=np.int64)
out = np.empty((1, 1), dtype=np.float32)
band.band_scores(read, one, np.array([4], dtype=np.int64), one, read[None], zero,
                 np.array([1.0], dtype=np.float32), np.array([-1.0], dtype=np.float32), 8, 4.0, -8.0, -1e9, out)
assert out[0, 0] == 4.0, out
