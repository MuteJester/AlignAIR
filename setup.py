"""Build the optional Cython extension. All package metadata lives in ``pyproject.toml``; this shim
exists only to declare + compile the germline-CIGAR kernel (``alignair.predict._derive_cy``).

The extension is *best-effort*: if Cython or a C compiler is unavailable, the build is skipped and the
package falls back to the identical pure-Python implementation at import time (see
``alignair.predict.heuristic_matcher.DERIVE_BACKEND``). Binary wheels built in CID (cibuildwheel) ship
the compiled kernel; a plain source install still works everywhere.
"""
from setuptools import Extension, setup
from setuptools.command.build_ext import build_ext


class OptionalBuildExt(build_ext):
    """Never fail the whole install if the optional C extension can't be compiled."""

    def run(self):
        try:
            super().run()
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: optional Cython extension not built ({exc}); using pure-Python fallback")

    def build_extension(self, ext):
        try:
            super().build_extension(ext)
        except Exception as exc:  # pragma: no cover
            print(f"WARNING: skipping optional extension {ext.name} ({exc}); using pure-Python fallback")


def _ext_modules():
    try:
        from Cython.Build import cythonize
    except Exception:  # pragma: no cover - Cython absent -> pure-Python fallback
        return []
    return cythonize(
        [Extension("alignair.predict._derive_cy", ["src/alignair/predict/_derive_cy.pyx"])],
        compiler_directives={"language_level": "3"},
    )


setup(ext_modules=_ext_modules(), cmdclass={"build_ext": OptionalBuildExt})
