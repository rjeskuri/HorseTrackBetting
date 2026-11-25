"""The rail selector against real JRA going pages (ticket #253): the synthetic markup of
tests/fixtures/synthetic/jra_going_*.html has not drifted from the structure the site serves.

Every test is marked `local` and uses `real_fixtures` (tests/conftest.py): without
$HORSETRACK_DATA_DIR/fixtures the file is skipped. The pages are
`$HORSETRACK_DATA_DIR/fixtures/jra/going_<YYYY-MM-DD>_<VENUE>.html` (VENUE a jra.yaml code, the
date the race date the page names, e.g. going_2026-11-28_TOK.html), saved UTF-8 with a leading
comment naming the URL and fetch time, as the other real JRA pages are; they are re-encoded to
cp932 here, as on the wire. Pick a turf meeting day whose page prints a course in use.

With the fixtures directory present but no such file, the test FAILS rather than skips: the real
page is this ticket's deliverable, and it must be fetched once (through the project's ingest,
cache-first, rate-limited; never committed). Expectations come from the file name and from the
page's own `.rail .value` text, read here with a plain regex, not from the parser.
"""

from __future__ import annotations

import re
import unicodedata
from datetime import date, datetime, timezone

import pytest
from bs4 import BeautifulSoup

from horsetrack.core.config import load_source
from horsetrack.countries.japan.jra import going

pytestmark = pytest.mark.local

NAME = re.compile(r"going_(\d{4}-\d{2}-\d{2})_([A-Z]{3})\.html")
FETCHED = datetime(2026, 12, 1, 3, 0, tzinfo=timezone.utc)


@pytest.fixture
def real_going(real_fixtures):
    files = sorted(
        p for p in (real_fixtures / "jra").glob("going_*.html") if NAME.fullmatch(p.name)
    )
    return files


def test_the_rail_selector_and_a_to_d_form_hold_on_real_going_pages(real_going):
    assert real_going, (
        "no real going page: save one as $HORSETRACK_DATA_DIR/fixtures/jra/"
        "going_<YYYY-MM-DD>_<VENUE>.html (fetched through horsetrack.core.ingest)"
    )
    venues = set(load_source("japan", "jra").venues)
    for path in real_going:
        day, venue = NAME.fullmatch(path.name).groups()
        body = path.read_text(encoding="utf-8").encode("cp932", errors="replace")

        # independent of the parser: the page prints the rail in `.rail .value` as '<A-D>コース'
        doc = BeautifulSoup(body.decode("cp932", errors="replace"), "lxml")
        node = doc.select_one(".rail .value")
        assert node is not None, f"{path.name}: no `.rail .value`: the selector has drifted"
        text = " ".join(unicodedata.normalize("NFKC", node.get_text(" ")).split())
        printed = re.fullmatch(r"([A-D])コース.*", text)
        assert printed, f"{path.name}: rail value {text!r} is not the A-D form"

        frame = going.parse_going_page(body, fetched_utc=FETCHED)
        assert venue in venues
        assert set(frame["venue"]) == {venue}, path.name
        assert set(frame["race_date"]) == {date.fromisoformat(day)}, path.name
        by_surface = dict(zip(frame["surface"], frame["rail_setting"], strict=True))
        assert by_surface["turf"] == printed.group(1), path.name
        assert all(v is None for s, v in by_surface.items() if s != "turf"), path.name
