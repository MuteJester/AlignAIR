"""The AlignAIR code the replay mirrors, pinned by digest, and the check that it has not changed.

``replay.replay`` calls AlignAIR's own stage functions, but it mirrors the orchestration of three product functions:
the stage order, the masks, the configuration and the batched forward pass of ``pipeline.predict``,
``api.predict_sequences`` and ``forward.run_model``. ``PINNED`` holds a digest of each one's code
(``code_digest``), taken when the replay was last checked against it; ``api.predict_sequences`` must also still run
``pipeline.predict`` (``api._predict``). ``pin_changes`` names whatever no longer holds, so a product edit can never
make the replay silently diverge from the product: the equivalence tests fail on it, and
``workflow.diagnose.evaluate`` refuses to run (``unverified_message``) unless told to run unverified.

To re-pin after an AlignAIR change: update ``diagnostics/replay.py`` to the new code, run the equivalence tests
(``tests/alignair_assay/diagnostics/test_replay.py``, with ``ALIGNAIR_ASSAY_MODEL`` set for the released
checkpoint), then put the new digests in ``PINNED``.
"""
from __future__ import annotations

import hashlib
import inspect
import io
import json
import textwrap
import tokenize

from alignair import api
from alignair.predict import forward, pipeline

# The digest of each mirrored product function's code when the replay was last checked against it.
PINNED = {
    "alignair.predict.pipeline.predict": "908c509699ed16b15a261194bd80e5b240bfc1b496824ba85cb773f99f9af6cc",
    "alignair.api.predict_sequences": "d7e7d8dc48c0d304a71cd05fe3671f02802da12f6976cceda9009a1a4a439469",
    "alignair.predict.forward.run_model": "b0f4ef3f9a878d87f630bd26018bc34f5031b90849901a948ed50383e50d27ac",
}
# The name ``api.predict_sequences`` runs the pipeline through, which must be ``pipeline.predict``.
PREDICT_ALIAS = "alignair.api._predict"
# What ``pin_changes`` gives for the alias when it is pipeline.predict (it is never a change).
PREDICT_ALIAS_OK = "alignair.predict.pipeline.predict"


def mirrored() -> dict[str, object]:
    """The product functions the replay mirrors, by the names ``PINNED`` uses, as this process imported them."""
    return {"alignair.predict.pipeline.predict": pipeline.predict,
            "alignair.api.predict_sequences": api.predict_sequences,
            "alignair.predict.forward.run_model": forward.run_model}


def code_digest(function) -> str:
    """SHA-256 of the function's tokens as (type name, text), comments and blank lines left out. Token type numbers
    differ between Python versions (3.12 added EXCLAMATION, moving OP from 54 to 55), so the name is hashed. Python
    3.12 also tokenizes f-strings differently (FSTRING_START/MIDDLE/END instead of one STRING), so a pinned function
    that gains an f-string gets a different digest per version; none of them has one."""
    source = textwrap.dedent(inspect.getsource(function))
    tokens = tokenize.generate_tokens(io.StringIO(source).readline)
    kept = [(tokenize.tok_name[tok.type], tok.string) for tok in tokens
            if tok.type not in (tokenize.COMMENT, tokenize.NL)]
    return hashlib.sha256(repr(kept).encode()).hexdigest()


def current_digests() -> dict[str, str]:
    """The current ``code_digest`` of each mirrored function, and what ``api._predict`` is (``PREDICT_ALIAS``):
    ``PREDICT_ALIAS_OK`` when it is ``pipeline.predict``, else the qualified name of what it is now."""
    digests = {name: code_digest(function) for name, function in mirrored().items()}
    alias = api._predict
    digests[PREDICT_ALIAS] = (PREDICT_ALIAS_OK if alias is pipeline.predict else
                              f"{getattr(alias, '__module__', '?')}.{getattr(alias, '__qualname__', repr(alias))}")
    return digests


def pin_changes(digests: dict[str, str] | None = None) -> dict[str, str]:
    """Name -> new digest of each mirrored function whose code is not the pinned one, and ``PREDICT_ALIAS`` -> what
    it is now when ``api._predict`` is no longer ``pipeline.predict``; empty when the replay is verified.
    ``digests`` is ``current_digests()`` when given."""
    digests = current_digests() if digests is None else digests
    changed = {name: digests[name] for name in PINNED if digests[name] != PINNED[name]}
    if digests[PREDICT_ALIAS] != PREDICT_ALIAS_OK:
        changed[PREDICT_ALIAS] = digests[PREDICT_ALIAS]
    return changed


def replay_digest(digests: dict[str, str] | None = None) -> str:
    """SHA-256 of ``current_digests()`` (or the given ``digests``): the product code the replay runs against, for
    cache keys."""
    digests = current_digests() if digests is None else digests
    return hashlib.sha256(json.dumps(digests, sort_keys=True).encode()).hexdigest()


def unverified_message(changes: dict[str, str]) -> str:
    """Why a replay against ``changes`` (``pin_changes``) is unverified, and how to re-pin or run anyway."""
    names = "; ".join(f"{name} (now {digest})" for name, digest in changes.items())
    return (f"the AlignAIR code the diagnostics replay mirrors has changed since the replay was pinned, so the replay "
            f"may no longer equal the product pipeline: {names}. To re-pin: update diagnostics/replay.py to the new "
            "code, run the equivalence tests (tests/alignair_assay/diagnostics/test_replay.py), then update PINNED "
            "in diagnostics/pin.py. To diagnose anyway, unverified: allow_unverified_replay=True (diagnose "
            "--allow-unverified-replay)")
