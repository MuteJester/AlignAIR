"""The README and the user's pages of the website speak to a repertoire researcher: no words of the model's internals
(those live on "How AlignAIR works" and the developer pages), nothing saying AlignAIR 4 or its models are not out yet
(the docs are written for release day), and every section heading has an id, so the outline and heading links work.

Pages not written yet are strict expected failures: each one leaves ``_NOT_YET`` when it is written, and a listed file
that passes fails the test, so the list only shrinks."""
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "website" / "src" / "pages" / "docs"
USER_FILES = ["README.md", "website/src/pages/Home.tsx", "website/src/pages/docs/pages-start.tsx",
              "website/src/pages/docs/pages-coming-from.tsx", "website/src/pages/docs/pages-use.tsx",
              "website/src/pages/docs/pages-results.tsx", "website/src/pages/docs/pages-reference.tsx",
              "website/src/pages/docs/pages-train.tsx", "website/src/pages/docs/pages-project.tsx"]
_NOT_YET = set(USER_FILES) - {"README.md", "website/src/pages/Home.tsx", "website/src/pages/docs/pages-start.tsx",
                               "website/src/pages/docs/pages-coming-from.tsx", "website/src/pages/docs/pages-use.tsx"}

# The model's internals; "How AlignAIR works" (slug how-alignair-works) may name them.
INTERNAL = re.compile(r"\b(neural|chip|trunk|heads|bank|forward pass|embeddings?|curriculum|logits?)\b", re.IGNORECASE)
NOT_OUT_YET = re.compile(r"unreleased|not (yet )?published|not yet released|PENDING-V4", re.IGNORECASE)
COMMENTS = re.compile(r"\{/\*.*?\*/\}|<!--.*?-->|/\*.*?\*/", re.DOTALL)
_PAGE = re.compile(r'slug:\s*"([^"]+)"')


def _without_the_internals_page(text: str) -> str:
    """The text without the ``how-alignair-works`` page object (it may name the internals)."""
    starts = [m.start() for m in _PAGE.finditer(text)] + [len(text)]
    return "".join(text[a:b] for a, b in zip(starts, starts[1:]) if _PAGE.match(text[a:b]).group(1)
                   != "how-alignair-works") if starts[:-1] else text


def _param(name: str):
    return pytest.param(name, marks=pytest.mark.xfail(strict=True, reason="not written yet")) \
        if name in _NOT_YET else name


@pytest.mark.parametrize("name", [_param(name) for name in USER_FILES])
def test_the_user_path_uses_plain_words_and_is_written_for_release_day(name):
    text = (ROOT / name).read_text()
    visible = COMMENTS.sub("", _without_the_internals_page(text))
    assert sorted({m.group(0).lower() for m in INTERNAL.finditer(visible)}) == []
    assert NOT_OUT_YET.findall(COMMENTS.sub("", text)) == []
    assert re.findall(r"<h[23][\s>]", text) == []          # headings through H2/H3 with an id


def test_the_checks_catch_what_they_guard_against():
    assert INTERNAL.search("the chip reads the bank") and not INTERNAL.search("a headline about banking")
    assert NOT_OUT_YET.search("models are not yet published") and not NOT_OUT_YET.search(COMMENTS.sub(
        "", "{/* PENDING-V4-PUBLISH: the id */} alignair4-igh-human"))
