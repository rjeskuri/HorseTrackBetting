"""The Hong Kong model card's status line (ticket #67, outline step 3).

Under `## Status` the card carries exactly one status line, dated:
`Status: data ported, otherwise paused (YYYY-MM-DD)`, the date a real calendar date not after today.
The exact date is the coder's to write; only its shape and plausibility are pinned here.
"""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import horsetrack.countries.hongkong as hongkong

CARD = Path(hongkong.__file__).parent / "MODEL_CARD.md"
STATUS_LINE = re.compile(r"^Status: data ported, otherwise paused \((\d{4}-\d{2}-\d{2})\)$")


def _status_section(text: str) -> list[str]:
    lines = text.splitlines()
    starts = [i for i, line in enumerate(lines) if line.strip() == "## Status"]
    assert len(starts) == 1, f"expected one '## Status' heading, found {len(starts)}"
    body: list[str] = []
    for line in lines[starts[0] + 1 :]:
        if line.startswith("## "):
            break
        body.append(line.strip())
    return body


def test_status_section_has_one_dated_status_line() -> None:
    section = _status_section(CARD.read_text(encoding="utf-8"))
    status_lines = [line for line in section if line.startswith("Status:")]
    assert len(status_lines) == 1, f"expected exactly one status line, found {status_lines!r}"
    match = STATUS_LINE.match(status_lines[0])
    assert match, f"status line is not the dated 'data ported' line: {status_lines[0]!r}"
    stamped = date.fromisoformat(match.group(1))
    assert stamped <= date.today(), f"status date {stamped} is in the future"
