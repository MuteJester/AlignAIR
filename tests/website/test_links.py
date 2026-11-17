"""Every link on the website leads somewhere: a ``DocLink``, ``/docs/<slug>`` or ``#/docs/<slug>`` target is a page of
the docs registry, and a ``?h=<id>`` heading target is a heading of that page; the search index links only to those;
and every address retired in the restructure (nav.ts ``MOVED``) redirects to a live page."""
import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.component

SRC = Path(__file__).resolve().parents[2] / "website" / "src"
_PAGE = re.compile(r'slug:\s*"([^"]+)"')
_HEADING_ID = re.compile(r'<H[23]\s+id="([^"]+)"')
_LINKS = re.compile(r'DocLink\s+to="([^"]+)"|to[=:]\s*"/docs/([^"]+)"|href="#/docs/([^"]+)"')


def _data(name: str):
    return json.loads((SRC / "data" / name).read_text())


def _table_anchors(chunk: str) -> set[str]:
    """The row ids of the generated tables a page renders (each row of OptionTable, ColumnTable, ModelTable, ErrorTable
    and PythonApi is a link target, named in the data file's ``anchor``)."""
    anchors = set()
    options = _data("cli-options.json")
    for command in re.findall(r'<OptionTable\s+command="([^"]+)"', chunk):
        anchors |= {option["anchor"] for group in options[command]["groups"] for option in group["options"]}
    for named, names in re.findall(r"<ColumnTable(\s+names=\{(.*?)\})?\s*/>", chunk, re.DOTALL):
        listed = set(re.findall(r'"([^"]+)"', names))    # names held in a variable: no row counted
        anchors |= {column["name"] for column in _data("output-columns.json") if not named or column["name"] in listed}
    if "<ModelTable" in chunk:
        anchors |= {model["anchor"] for model in _data("models.json")}
    if "<ErrorTable" in chunk:
        anchors |= {error["anchor"] for error in _data("error-codes.json")}
    for package in re.findall(r'<PythonApi\s+packageName="([^"]+)"', chunk):
        anchors |= {entry["anchor"] for entry in _data("python-api.json") if entry["name"].rpartition(".")[0] == package}
    return anchors


def _pages() -> dict[str, set[str]]:
    """Each docs page's slug and its link targets: heading ids and the rows of the tables it renders."""
    pages = {}
    for source in sorted((SRC / "pages" / "docs").glob("pages-*.tsx")):
        text = source.read_text()
        starts = [match.start() for match in _PAGE.finditer(text)] + [len(text)]
        for begin, end in zip(starts, starts[1:]):
            chunk = text[begin:end]
            pages[_PAGE.match(chunk).group(1)] = set(_HEADING_ID.findall(chunk)) | _table_anchors(chunk)
    return pages


def _broken(target: str, pages: dict[str, set[str]]) -> bool:
    slug, _, heading = target.partition("?h=")
    return slug not in pages or bool(heading) and heading not in pages[slug]


def _moved() -> dict[str, str]:
    """nav.ts's ``MOVED`` table: each retired slug and the slug that replaced it."""
    text = (SRC / "pages" / "docs" / "nav.ts").read_text()
    block = text[text.index("export const MOVED"):]
    block = block[:block.index("};")]
    return dict(re.findall(r'"?([a-z0-9-]+)"?:\s*"([a-z0-9-]+)"', block.partition("= {")[2]))


def test_every_retired_address_redirects_to_a_live_page():
    pages, moved = _pages(), _moved()
    assert len(moved) > 20 and moved["getting-started"] == "first-alignment"
    assert sorted(slug for slug in moved.values() if slug not in pages) == []
    assert sorted(slug for slug in moved if slug in pages) == []


def test_every_site_link_leads_to_a_page_and_heading():
    pages = _pages()
    broken = [f"{source.relative_to(SRC)}: {target}" for source in sorted(SRC.rglob("*.tsx"))
              for groups in _LINKS.findall(source.read_text()) for target in groups if target
              and _broken(target, pages)]
    assert broken == []


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
    assert not _broken("cli?h=predict-model", pages) and _broken("cli?h=predict-no-such-option", pages)
