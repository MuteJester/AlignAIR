"""AIRR-review guard: every ``alignair …`` command shown in the public docs/examples must actually
parse against the real argparse CLI — not merely use known flag names. This runs the true
``parser.parse_args`` (which never executes the command), so it rejects positional-vs-``--input`` drift,
``-o``, missing required options, unknown subcommands, and missing nested subcommands. Prevents the
class of doc bug the reviewer found (``-o``/``--reference``/``alignair bundle``)."""
from __future__ import annotations

import contextlib
import io as _io
import re
import shlex
from pathlib import Path

import pytest

from alignair.cli.main import build_parser

_ROOT = Path(__file__).resolve().parents[3]

# public surfaces users copy-paste from (implementation/architecture notes are intentionally excluded)
_DOC_FILES = [
    _ROOT / "README.md",
    _ROOT / "docs" / "getting_started.md",
    _ROOT / "docs" / "python_api.md",
    *sorted((_ROOT / "examples").glob("**/README.md")),
]

# split a compound shell line into simple commands (so `cat x | alignair …` and `a && b` are handled)
_SHELL_SPLIT = re.compile(r"\s*(?:\|\||&&|\||;)\s*")
# shell redirect operators to strip (with their filename target, except the fd-dup forms)
_REDIRECT_WITH_TARGET = {">", ">>", "<", "2>", "1>", "&>", "&>>"}
_REDIRECT_BARE = {"2>&1", "1>&2", "&>&1"}


def _iter_shell_commands(text: str):
    """Yield each ``alignair …`` command from ```` ```bash/sh/console ```` fences, joining line
    continuations and splitting compound lines. A leading ``$ `` prompt is stripped."""
    for block in re.findall(r"```(?:bash|sh|console)\n(.*?)```", text, flags=re.DOTALL):
        joined = block.replace("\\\n", " ")
        for line in joined.splitlines():
            line = line.strip()
            if line.startswith("$ "):
                line = line[2:].strip()
            if not line or line.startswith("#"):
                continue
            for piece in _SHELL_SPLIT.split(line):
                piece = piece.strip()
                if piece.startswith("alignair "):
                    yield piece


def _strip_redirects(tokens) -> list:
    out, i = [], 0
    while i < len(tokens):
        t = tokens[i]
        if t in _REDIRECT_WITH_TARGET:
            i += 2                                 # skip the operator and its filename
            continue
        if t in _REDIRECT_BARE:
            i += 1
            continue
        out.append(t)
        i += 1
    return out


def _problems_for(cmd: str, parser) -> list:
    """Return a list of problems for one ``alignair …`` command (empty == it parses)."""
    try:
        tokens = shlex.split(cmd, comments=True)   # honor quotes; drop trailing `# …` comments
    except ValueError as e:
        return [f"unparseable shell command {cmd!r}: {e}"]   # never skip silently (reviewer)
    tokens = _strip_redirects(tokens[1:])          # drop leading 'alignair'
    buf = _io.StringIO()
    try:
        with contextlib.redirect_stderr(buf), contextlib.redirect_stdout(buf):
            parser.parse_args(tokens)              # parse only — argparse does NOT run the command
    except SystemExit as e:
        if e.code not in (0, None):                # 0 == a --help/--version style clean exit
            msg = (buf.getvalue().strip().splitlines() or ["parse error"])[-1]
            return [f"does not parse: `alignair {' '.join(tokens)}` -> {msg}"]
    return []


@pytest.mark.parametrize("doc", [d for d in _DOC_FILES if d.exists()], ids=lambda d: d.name)
def test_doc_alignair_commands_parse(doc):
    parser = build_parser()
    problems = []
    for cmd in _iter_shell_commands(doc.read_text()):
        problems += _problems_for(cmd, parser)
    assert not problems, "stale CLI in docs:\n" + "\n".join(problems)


# --- the guard itself must reject the known drift classes (reviewer's required regressions) ----------

@pytest.mark.parametrize("bad", [
    "alignair predict reads.fasta --model m.alignair --out o.tsv",   # positional input (no --input)
    "alignair predict --input r.fasta -o o.tsv --model m.alignair",  # -o instead of --out
    "alignair predict --out o.tsv --model m.alignair",               # missing --input
    "alignair predict --input r.fasta --model m.alignair",           # missing --out
    "alignair bundle --model ckpt.pt -o out/",                       # nonexistent subcommand
    "alignair models",                                               # missing nested subcommand
    "alignair train --dataconfig HUMAN_IGH_OGRDB --out r --preset standard",  # invalid choice
])
def test_guard_rejects_stale_commands(bad):
    assert _problems_for(bad, build_parser()), f"guard should have rejected: {bad}"


def test_guard_accepts_a_valid_command():
    assert not _problems_for(
        "alignair predict --input r.fasta --out o.tsv --model m.alignair", build_parser())
