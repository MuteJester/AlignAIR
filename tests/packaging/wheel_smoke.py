"""cibuildwheel smoke test: verify the compiled germline-CIGAR kernel loaded in the built wheel,
WITHOUT importing the full ``alignair`` package (which pulls heavy runtime deps like torch)."""
import glob
import importlib.util
import os
import sys

origin = importlib.util.find_spec("alignair").origin          # locate, don't execute (no torch import)
matches = glob.glob(os.path.join(os.path.dirname(origin), "predict", "_derive_cy*.*"))
so = [m for m in matches if m.endswith((".so", ".pyd"))]
if not so:
    sys.exit(f"compiled _derive_cy kernel missing from the wheel (found: {matches})")

spec = importlib.util.spec_from_file_location("_derive_cy", so[0])   # name must match PyInit__derive_cy
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
assert mod.derive_cigar(b"ACGGT", b"ACGT", 1) == "2M1I2M", mod.derive_cigar(b"ACGGT", b"ACGT", 1)
print("cython kernel OK:", os.path.basename(so[0]))
