"""Every link on the website leads somewhere: a ``DocLink``, ``/docs/<slug>`` or ``#/docs/<slug>`` target is a page of
the docs registry, and a ``?h=<id>`` heading target is a heading of that page; the search index links only to those."""
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

SRC = Path(__file__).resolve().parents[2] / "website" / "src"
_PAGE = re.compile(r'slug:\s*"([^"]+)"')
_HEADING_ID = re.compile(r'<H[23]\s+id="([^"]+)"')
_LINKS = re.compile(r'DocLink\s+to="([^"]+)"|to[=:]\s*"/docs/([^"]+)"|href="#/docs/([^"]+)"')


def _pages() -> dict[str, set[str]]:
    """Each docs page's slug and its heading ids."""
    pages = {}
    for source in sorted((SRC / "pages" / "docs").glob("pages-*.tsx")):
        text = source.read_text()
        starts = [match.start() for match in _PAGE.finditer(text)] + [len(text)]
        for begin, end in zip(starts, starts[1:]):
            chunk = text[begin:end]
            pages[_PAGE.match(chunk).group(1)] = set(_HEADING_ID.findall(chunk))
    return pages


def _broken(target: str, pages: dict[str, set[str]]) -> bool:
    slug, _, heading = target.partition("?h=")
    return slug not in pages or bool(heading) and heading not in pages[slug]


def test_every_site_link_leads_to_a_page_and_heading():
    pages = _pages()
    broken = [f"{source.relative_to(SRC)}: {target}" for source in sorted(SRC.rglob("*.tsx"))
              for groups in _LINKS.findall(source.read_text()) for target in groups if target
              and _broken(target, pages)]
    assert broken == []


@pytest.mark.xfail(strict=True, reason="the reference pages the index links to are written in the docs part's group 5")
def test_every_search_result_leads_to_a_page_and_heading():
    pages = _pages()
    entries = json.loads((SRC / "data" / "search-index.json").read_text())
    assert [entry["href"] for entry in entries if _broken(entry["href"].removeprefix("#/docs/"), pages)] == []


def test_the_link_checker_finds_links_and_reports_a_broken_one():
    pages = _pages()
    assert len(pages) > 20
    found = [target for source in SRC.rglob("*.tsx") for groups in _LINKS.findall(source.read_text())
             for target in groups if target]
    assert len(found) > 20
    assert _broken("no-such-page", pages) and _broken(f"{next(iter(pages))}?h=no-such-heading", pages)
