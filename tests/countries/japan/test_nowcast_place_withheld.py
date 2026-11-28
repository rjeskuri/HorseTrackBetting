"""#410 round 7, finding 2 (owner's restart, 2026-11-28): the committed docs state no outcome of
the JRA-VAN Place measurement in words (D21).

The v5 measurement reads JRA-VAN boards against JRA-VAN votes, so whether a read stands as a
result discloses a bound on licensed data (`PLACE_MIN_WITHIN_TOLERANCE` of the validated boards
within the tolerance). Contract pinned here, on the docs as committed:

- Scope: the report's `## #410` section (docs/experiments/nowcast-japan.md, up to the next `## `
  heading), the ledger's #410 rows (docs/ledger.md, table rows starting `| #410`), and the lines
  naming #410 in src/horsetrack/countries/japan/MODEL_CARD.md and docs/modeling.md.
- Text is read without backticks or bold markers, whitespace collapsed, case ignored, split into
  sentences at `.`/`!`/`?` before whitespace.
- No sentence in scope names a Place read (`joint`, `low_only`, "the other read(s)", "the
  others") together with an outcome in words ("stands", "meets", "passed", "as a result", "is/are
  (not) a result", "is/are a diagnostic"), unless it is a stated rule rather than an outcome: it
  says "when", "if", "unless", "whether", "whatever" or "never". No sentence in scope says the
  measurement "meets the criterion" or "meets DoD".
- The report's #410 section and the ledger's #410 row say "withheld (D21), see the private report"
  (the #408 wording), and still describe what was measured (`place_rule_measurement`, the `joint`
  read and its fit `place_money_fit`).
"""

from __future__ import annotations

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
REPORT = REPO_ROOT / "docs/experiments/nowcast-japan.md"
LEDGER = REPO_ROOT / "docs/ledger.md"
MODEL_CARD = REPO_ROOT / "src/horsetrack/countries/japan/MODEL_CARD.md"
MODELING = REPO_ROOT / "docs/modeling.md"

WITHHELD = "withheld (d21), see the private report"
READ = re.compile(r"\b(?:joint|low_only)\b|\bother reads?\b|\bthe others\b")
OUTCOME = re.compile(
    r"\b(?:stands?|stood|meets?|met|pass(?:es|ed)|as a result"
    r"|(?:is|are) (?:not )?a (?:result|diagnostic))\b"
)
RULE = re.compile(r"\b(?:when|if|unless|whether|whatever|never)\b")
CRITERION = re.compile(r"\b(?:meets?|met) (?:the |its )?(?:\S+ )?(?:criterion|dod)\b")


def normal(text: str) -> str:
    return re.sub(r"\s+", " ", text.replace("`", "").replace("**", "")).strip().lower()


def report_section() -> str:
    lines = REPORT.read_text(encoding="utf-8").splitlines()
    starts = [i for i, line in enumerate(lines) if line.startswith("## #410")]
    assert len(starts) == 1, f"{len(starts)} '## #410' headings in {REPORT.name}"
    end = next(
        (i for i in range(starts[0] + 1, len(lines)) if lines[i].startswith("## ")), len(lines)
    )
    return "\n".join(lines[starts[0] : end])


def ledger_rows() -> list[str]:
    rows = [r for r in LEDGER.read_text(encoding="utf-8").splitlines() if r.startswith("| #410")]
    assert rows, f"no '| #410' row in {LEDGER.name}"
    return rows


def lines_naming_410(path: Path) -> str:
    return "\n".join(
        line for line in path.read_text(encoding="utf-8").splitlines() if "#410" in line
    )


def scoped() -> dict[str, str]:
    return {
        "report #410 section": report_section(),
        "ledger #410 row": "\n".join(ledger_rows()),
        "MODEL_CARD #410 lines": lines_naming_410(MODEL_CARD),
        "modeling.md #410 lines": lines_naming_410(MODELING),
    }


def verdicts(text: str) -> list[str]:
    """The sentences of `text` that state an outcome of the Place measurement in words."""
    out = []
    for sentence in re.split(r"(?<=[.!?])\s+", normal(text)):
        if CRITERION.search(sentence) or (
            READ.search(sentence) and OUTCOME.search(sentence) and not RULE.search(sentence)
        ):
            out.append(sentence)
    return out


def assert_rules_hold() -> None:
    """Precondition: the rules flag the outcome forms the round-7 review named, and leave a stated
    rule or the withheld form alone (made-up sentences, no figure)."""
    flagged = [
        "The `joint` read stands as a result: it meets the criterion stated above.",
        "Its measurement stands the `joint` read as a result.",
        "The other reads (`low_only`, `undetermined`) are not a result.",
        "Boards read undetermined are a diagnostic, as are `low_only` boards.",
        "Measured in v5: the joint read stands, the others are a diagnostic.",
        "The private report reads the Place rows as a result, which meets DoD (b) for them.",
        "The **joint** read passed.",
    ]
    kept = [
        "A determined read (`joint` or `low_only`) stands as a result only when it is within.",
        "It is a diagnostic whatever its measurement: it never stands as a result.",
        "Whether the `joint` read stands as a result: withheld (D21), see the private report.",
        "`joint`, the board path wherever a board prints a high figure on every runner.",
        "The measurement validates the boards the study scores.",
    ]
    for sentence in flagged:
        assert verdicts(sentence), sentence
    for sentence in kept:
        assert not verdicts(sentence), sentence


def test_the_place_measurement_outcome_is_withheld_from_the_committed_docs():
    """Round-7 finding 2: no committed doc states the JRA-VAN Place measurement's outcome in words
    (which read stands as a result, that it meets the criterion); the report's #410 section and
    the #410 ledger row say it is withheld (D21) with the pointer to the private report, and keep
    the verdict-free description of what was measured."""
    assert_rules_hold()
    found = {name: verdicts(text) for name, text in scoped().items()}
    stated = {name: hits for name, hits in found.items() if hits}
    assert not stated, "outcome stated in words (D21):\n" + "\n".join(
        f"- {name}: {hit[:160]}" for name, hits in stated.items() for hit in hits
    )
    for name, text in (
        ("report #410 section", report_section()),
        ("ledger #410 row", "\n".join(ledger_rows())),
    ):
        text = normal(text)
        assert WITHHELD in text, f"{name} does not say '{WITHHELD}'"
        for kept in ("joint", "place_money_fit"):
            assert kept in text, f"{name} no longer describes the measurement ({kept})"
    assert "place_rule_measurement" in normal(report_section())
