"""Ticket #74: the append-only experiment registry and its committed summary.

Contract pinned here (the coder implements what these tests call):

- ``FoldMetric(fold, metric, value, lo, hi, n)`` and ``RunRecord(hypothesis_id, family_id,
  variant, ticket, git_sha, data_builds, sources, fold_scheme, components, params, metrics,
  status, holdout_open=None, owner_signoff=None)`` compare equal by value. ``records()`` gives
  back equal objects (tuples stay tuples, floats are exact) in append order, from any
  ``Registry`` opened on the same file; a registry with no file yet has no records.
- ``Registry(path, *, holdout_log)`` keeps one JSON line per run in ``path``. An append only adds
  bytes at the end (the file keeps its inode), one line per run, whichever ``Registry`` object on
  the file makes it; each line carries the sha256 hex of the previous line. No public name on
  ``Registry`` updates or deletes. Appending an equal record twice is a duplicate run id and is
  refused.
- ``verify()`` raises ``RegistryTampered`` (never a decode error) when any byte of any line is
  changed, the last line included, and when a line other than the last is deleted, swapped or
  replayed. It passes on an untouched file, and again once the file is restored.
- Refusals raise ``RecordRefused`` and write nothing: a hypothesis id that is not ``E``, ``L`` or
  ``P`` followed by ASCII digits; a blank family id; a status other than failed, scored, rejected
  or adopted; a ``holdout_open`` that is not the sha256 hex of a line (its bytes without the
  newline) of ``holdout_log`` as that file is at append time; ``adopted`` without a positive int
  ``owner_signoff``.
- ``family_size`` counts the family's distinct variants, a variant being the run's specification
  ``(hypothesis_id, variant, fold_scheme, sources, components, params)`` plus the feature-table
  identity of its ``data_builds``, with mappings compared whatever their key order: every variant
  tried counts once whatever its status and however many runs it has, failed ones included. A
  ``data_builds`` value ``"<identity>@<refresh>"`` splits at its last ``@``: the build name and
  the identity part name the feature table, the refresh id does not; a value without ``@`` is
  identity in full. A change to params, components, fold_scheme, sources or a build's identity,
  or an added, removed or renamed build, is a new variant even under the same label. A later
  decision, a retry, a rerun at another git_sha or ticket, or a refresh of the same builds is
  another run (``records()`` keeps every run), not another variant.
- ``freeze_family(family_id, spec)`` returns
  ``sha256(json.dumps(spec, sort_keys=True, separators=(",", ":")))`` as hex. The freeze persists
  in the registry and is not a run; a different spec raises ``FamilyAlreadyFrozen``.
- ``write_summary(registry, directory)`` writes one file per family (``<family_id>.md``) plus an
  index ``README.md``; the family files hold the family, E id, variant, status and per-fold metric,
  value, CI and n of each run, and nothing else from the record. A run with a source in
  ``WITHHELD_SOURCES`` (in any spelling) shows ``withheld (licensed source)`` in place of its
  values, or is refused at append.
- ``default_registry(country)`` stores at ``data_dir()/registry/<country>.jsonl`` and creates the
  directory.

Every path is under ``tmp_path``; HORSETRACK_DATA_DIR points into ``tmp_path`` and an autouse
guard fails any test that changes a real ``src/horsetrack/countries/*/holdout_log.jsonl``. Field
names on disk, the run-id format and the summary's layout are left to the implementation. Every
value here is synthetic.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from horsetrack.core.eval.registry import (
    FamilyAlreadyFrozen,
    FoldMetric,
    RecordRefused,
    Registry,
    RegistryTampered,
    RunRecord,
    default_registry,
    write_summary,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
REAL_COUNTRIES = REPO_ROOT / "src" / "horsetrack" / "countries"
DATA_DIR_VAR = "HORSETRACK_DATA_DIR"

GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
BUILD_HASH = hashlib.sha256(b"synthetic build").hexdigest()
SEED_SHA = hashlib.sha256(b"synthetic seed").hexdigest()
WITHHELD_LABEL = "withheld (licensed source)"


# --------------------------------------------------------------------------- helpers


def fold(name: str, metric: str, value: float, lo, hi, n: int) -> FoldMetric:
    return FoldMetric(fold=name, metric=metric, value=value, lo=lo, hi=hi, n=n)


def make_record(**overrides) -> RunRecord:
    fields = {
        "hypothesis_id": "E1",
        "family_id": "F1",
        "variant": "baseline",
        "ticket": 74,
        "git_sha": GIT_SHA,
        "data_builds": {"results": BUILD_HASH},
        "sources": ("netkeiba",),
        "fold_scheme": "walk-forward-6",
        "components": {"model": "logit-1.0"},
        "params": {"C": 1.0},
        "metrics": (fold("wf-1", "log_loss", 0.69, 0.65, 0.73, 500),),
        "status": "scored",
    }
    fields.update(overrides)
    return RunRecord(**fields)


def registry_path(tmp_path: Path) -> Path:
    path = tmp_path / "data" / "registry" / "japan.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def holdout_log_path(tmp_path: Path) -> Path:
    path = tmp_path / "countries" / "japan" / "holdout_log.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def open_registry(tmp_path: Path) -> tuple[Registry, Path, Path]:
    path, log = registry_path(tmp_path), holdout_log_path(tmp_path)
    return Registry(path, holdout_log=log), path, log


def append_holdout_opens(log: Path, purposes: list[str], *, first_ticket: int = 71) -> list[str]:
    """Synthetic #71 log lines, written the way ``LockedHoldout.open`` writes them."""
    lines = []
    for i, purpose in enumerate(purposes):
        record = {
            "utc": f"2026-12-{i + 1:02d}T09:{i:02d}:00+00:00",
            "ticket": first_ticket + i,
            "purpose": purpose,
            "git_sha": GIT_SHA,
            "n_rows": 1200 + 37 * i,
            "period": {"start": "2026-11-28", "seed_sha256": SEED_SHA},
        }
        lines.append(json.dumps(record))
    with open(log, "a", encoding="utf-8") as handle:
        handle.write("".join(line + "\n" for line in lines))
    return lines


def line_sha(line: str) -> str:
    return hashlib.sha256(line.encode("utf-8")).hexdigest()


def file_bytes(path: Path) -> bytes | None:
    return path.read_bytes() if path.exists() else None


def line_spans(data: bytes) -> list[tuple[int, int]]:
    """(start, end) of each line, the newline excluded; the data ends with a newline."""
    spans, start = [], 0
    while start < len(data):
        end = data.index(b"\n", start)
        spans.append((start, end))
        start = end + 1
    return spans


def flip(data: bytes, pos: int) -> bytes:
    """One byte changed (low bit flipped): it either breaks the JSON or changes a value."""
    return data[:pos] + bytes([data[pos] ^ 0x01]) + data[pos + 1 :]


def expect_tampered(registry: Registry, what: str) -> None:
    try:
        registry.verify()
    except RegistryTampered:
        return
    pytest.fail(f"verify() did not raise RegistryTampered after {what}")


def summary_of(registry: Registry, tmp_path: Path) -> str:
    """The summary as written for ``registry``: every family file's text, joined (the index
    ``README.md`` is not a family, so it is left out)."""
    out = tmp_path / "docs" / "experiments" / "registry" / "japan"
    out.mkdir(parents=True, exist_ok=True)
    write_summary(registry, out)
    families = sorted(p for p in out.glob("*.md") if p.name != "README.md")
    return "\n".join(p.read_text(encoding="utf-8") for p in families)


def shows_number(text: str, number: str) -> bool:
    """Whether ``number`` (or a longer rounding of it) appears as a number of its own.

    A decimal may be followed by more digits (0.12 matches 0.123456); an integer may carry a
    thousands separator. Digits inside a longer number, a timestamp or a hash do not count.
    """
    if "." in number:
        pattern = r"(?<![0-9A-Za-z.])" + re.escape(number)
    else:
        forms = {number, f"{int(number):,}"}
        pattern = r"(?<![0-9A-Za-z.,])(?:" + "|".join(map(re.escape, forms)) + r")(?![0-9])"
    return re.search(pattern, text) is not None


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    """No test reads the real data directory or changes a real holdout log."""
    monkeypatch.setenv(DATA_DIR_VAR, str(tmp_path / "guard-data"))

    def snapshot() -> dict[Path, bytes]:
        return {p: p.read_bytes() for p in REAL_COUNTRIES.glob("*/holdout_log.jsonl")}

    before = snapshot()
    yield
    assert snapshot() == before, "a test wrote to a real holdout_log.jsonl"


# --------------------------------------------------------------------------- round trip


def varied_records(holdout_hash: str) -> list[RunRecord]:
    return [
        make_record(
            hypothesis_id="E1",
            family_id="F-win",
            variant="v-logit",
            params={
                "C": 0.5,
                "penalty": "l2",
                "features": ["odds", "draw", 3],
                "nested": {"depth": 3, "flag": True, "none": None},
            },
            metrics=(
                fold("wf-2021", "log_loss", 0.1 + 0.2, 0.25, 0.35, 812),
                fold("wf-2022", "brier", 1e-300, None, None, 3),
            ),
        ),
        make_record(
            hypothesis_id="L7",
            family_id="L-calib",
            variant="v-failed",
            ticket=88,
            data_builds={},
            sources=(),
            fold_scheme="walk-forward-4",
            components={},
            params={},
            metrics=(),
            status="failed",
        ),
        make_record(
            hypothesis_id="P12",
            family_id="P-edge",
            variant="v-adopted",
            ticket=141,
            git_sha="a" * 40,
            data_builds={"results": BUILD_HASH, "odds": hashlib.sha256(b"odds").hexdigest()},
            sources=("netkeiba", "jra"),
            components={"model": "gbm-2.1", "calibration": "isotonic-0.3"},
            params={"rounds": 400, "eta": 0.05},
            metrics=(fold("holdout", "log_loss", 0.6825, 0.6702, 0.6951, 20_413),),
            status="adopted",
            holdout_open=holdout_hash,
            owner_signoff=141,
        ),
        make_record(
            hypothesis_id="E1",
            family_id="F-win",
            variant="ペース-v2",
            params={"window": "2019-2023", "weights": [0.25, 0.75]},
            metrics=(fold("wf-2023", "roi", -2.5, -3.75, -1.25, 1_000_000),),
            status="rejected",
        ),
    ]


def test_an_appended_record_reads_back_unchanged(tmp_path):
    reg, path, log = open_registry(tmp_path)
    assert reg.records() == []
    reg.verify()  # no file yet: nothing to fail

    (opened,) = append_holdout_opens(log, ["final evaluation"])
    records = varied_records(line_sha(opened))
    ids = [reg.append(record) for record in records]

    assert all(isinstance(run_id, str) and run_id for run_id in ids)
    assert len(set(ids)) == len(ids)
    assert reg.records() == records

    reopened = Registry(path, holdout_log=log)
    got = reopened.records()
    assert got == records
    for record in got:
        assert isinstance(record, RunRecord)
        assert isinstance(record.sources, tuple)
        assert isinstance(record.metrics, tuple)
        assert all(isinstance(m, FoldMetric) for m in record.metrics)
    assert got[0].metrics[0].value == 0.1 + 0.2
    assert got[0].metrics[1].value == 1e-300
    assert got[2].holdout_open == line_sha(opened)
    assert got[2].owner_signoff == 141

    assert reopened.records("F-win") == [records[0], records[3]]
    assert reopened.records("P-edge") == [records[2]]
    assert reopened.records("no-such-family") == []
    reopened.verify()


# --------------------------------------------------------------------------- append only


FORBIDDEN_WORDS = (
    "update",
    "delete",
    "remove",
    "edit",
    "replace",
    "rewrite",
    "overwrite",
    "amend",
    "modify",
    "erase",
    "truncate",
    "drop",
    "purge",
    "discard",
    "clear",
    "pop",
    "set_",
)


def test_records_are_append_only(tmp_path):
    reg, path, log = open_registry(tmp_path)
    first = make_record(variant="first", params={"seed": 1})
    reg.append(first)
    previous = path.read_bytes()
    assert previous.endswith(b"\n")
    first_line = previous.split(b"\n")[0]
    inode = path.stat().st_ino

    # Separate processes each open their own Registry on the same file.
    other = Registry(path, holdout_log=log)
    writers = [other, reg, other, reg, Registry(path, holdout_log=log)]
    for n, writer in enumerate(writers, start=2):
        writer.append(make_record(variant=f"later-{n}", params={"seed": n}))
        current = path.read_bytes()
        assert current.startswith(previous), f"append {n} changed earlier bytes"
        added = current[len(previous) :]
        assert added.endswith(b"\n") and added.count(b"\n") == 1, "one run is one line"
        assert path.stat().st_ino == inode, "the file was replaced, not appended to"
        previous = current

    assert previous.split(b"\n")[0] == first_line
    fresh = Registry(path, holdout_log=log)
    fresh.verify()
    expected = ["first"] + [f"later-{n}" for n in range(2, 2 + len(writers))]
    assert [r.variant for r in fresh.records()] == expected

    # Each line carries the sha256 of the line before it.
    lines = previous.split(b"\n")[:-1]
    for before, line in zip(lines, lines[1:], strict=False):
        digests = {
            hashlib.sha256(before).hexdigest().encode(),
            hashlib.sha256(before + b"\n").hexdigest().encode(),
        }
        assert any(d in line for d in digests), "a line does not carry the previous line's hash"

    # No update or delete path.
    for obj in (Registry, reg):
        offending = [
            name
            for name in dir(obj)
            if not name.startswith("_") and any(word in name.lower() for word in FORBIDDEN_WORDS)
        ]
        assert offending == [], f"Registry exposes {offending}"

    # The same run twice is a duplicate run id.
    before_dup = path.read_bytes()
    with pytest.raises(RecordRefused):
        reg.append(first)
    with pytest.raises(RecordRefused):
        Registry(path, holdout_log=log).append(make_record(variant="first", params={"seed": 1}))
    assert path.read_bytes() == before_dup


# --------------------------------------------------------------------------- tamper evidence


TAMPER_MARKERS = ["tamper-alpha", "tamper-bravo", "tamper-charlie", "tamper-delta"]


def tamper_registry(tmp_path: Path) -> tuple[Registry, Path, Path, bytes]:
    reg, path, log = open_registry(tmp_path)
    for i, marker in enumerate(TAMPER_MARKERS, start=1):
        reg.append(make_record(hypothesis_id=f"E{i}", variant=marker, params={"seed": i}))
    reg.verify()
    return reg, path, log, path.read_bytes()


def test_an_edited_line_is_detected(tmp_path):
    assert issubclass(RegistryTampered, ValueError)
    reg, path, log, pristine = tamper_registry(tmp_path)

    def check(data: bytes, what: str) -> None:
        path.write_bytes(data)
        expect_tampered(reg, what)
        expect_tampered(Registry(path, holdout_log=log), what)

    # One byte of a value: the last letter of each run's variant, line 1 first.
    for marker in TAMPER_MARKERS:
        pos = pristine.index(marker.encode()) + len(marker) - 1
        check(flip(pristine, pos), f"editing the variant {marker!r}")

    # One byte anywhere in a line: line 1 and every other line, the last one included.
    spans = line_spans(pristine)
    assert len(spans) >= len(TAMPER_MARKERS)
    for number, (start, end) in enumerate(spans, start=1):
        step = max(1, (end - start) // 25)
        for pos in sorted({start, end - 1, *range(start, end, step)}):
            check(flip(pristine, pos), f"changing byte {pos - start} of line {number}")

    path.write_bytes(pristine)
    reg.verify()
    Registry(path, holdout_log=log).verify()


def test_a_deleted_or_reordered_line_is_detected(tmp_path):
    _, path, log, pristine = tamper_registry(tmp_path)
    lines = pristine.split(b"\n")[:-1]
    cases = {
        "deleting line 1": lines[1:],
        "deleting a middle line": lines[:1] + lines[2:],
        "swapping two lines": [lines[0], lines[2], lines[1], *lines[3:]],
        "replaying a line": [lines[0], lines[1], lines[1], *lines[2:]],
    }
    for what, kept in cases.items():
        path.write_bytes(b"".join(line + b"\n" for line in kept))
        expect_tampered(Registry(path, holdout_log=log), what)


# --------------------------------------------------------------------------- refusals


HYPOTHESIS_IDS = [
    *[
        pytest.param(hid, True, id=f"ok-{hid}")
        for hid in ("E1", "L2", "P3", "E10", "L042", "P999", "E0")
    ],
    *[
        pytest.param(hid, False, id=f"bad-{label}")
        for label, hid in (
            ("empty", ""),
            ("X1", "X1"),
            ("E-alone", "E"),
            ("L-alone", "L"),
            ("P-alone", "P"),
            ("lower-e1", "e1"),
            ("lower-p4", "p4"),
            ("E1a", "E1a"),
            ("1E", "1E"),
            ("E-dash-1", "E-1"),
            ("E-space-1", "E 1"),
            ("leading-space", " E1"),
            ("trailing-space", "E1 "),
            ("trailing-newline", "E1\n"),
            ("EE1", "EE1"),
            ("decimal", "E1.5"),
            ("fullwidth-digit", "E１"),
            ("cyrillic-E", "Е1"),
        )
    ],
]


@pytest.mark.parametrize(("hypothesis_id", "accepted"), HYPOTHESIS_IDS)
def test_a_run_without_hypothesis_id_is_refused(tmp_path, hypothesis_id, accepted):
    assert issubclass(RecordRefused, ValueError)
    reg, path, _ = open_registry(tmp_path)
    reg.append(make_record(hypothesis_id="E1", variant="anchor"))
    before = path.read_bytes()

    if accepted:
        reg.append(make_record(hypothesis_id=hypothesis_id, variant="candidate"))
        assert [r.hypothesis_id for r in reg.records()] == ["E1", hypothesis_id]
        return
    with pytest.raises(RecordRefused):
        reg.append(make_record(hypothesis_id=hypothesis_id, variant="candidate"))
    assert path.read_bytes() == before
    assert [r.variant for r in reg.records()] == ["anchor"]


@pytest.mark.parametrize("hypothesis_id", [None, 1], ids=["none", "int"])
def test_a_run_with_a_non_string_hypothesis_id_is_refused(tmp_path, hypothesis_id):
    reg, path, _ = open_registry(tmp_path)
    with pytest.raises((RecordRefused, TypeError)):
        reg.append(make_record(hypothesis_id=hypothesis_id))
    assert file_bytes(path) in (None, b"")


@pytest.mark.parametrize(
    ("family_id", "accepted"),
    [
        pytest.param("F1", True, id="ok-F1"),
        pytest.param("P-roi-2026", True, id="ok-P-roi"),
        pytest.param("L1-calibration", True, id="ok-L1-calibration"),
        pytest.param("", False, id="empty"),
        pytest.param("   ", False, id="spaces"),
        pytest.param("\t\n", False, id="whitespace"),
        pytest.param(None, False, id="none"),
    ],
)
def test_a_run_without_family_id_is_refused(tmp_path, family_id, accepted):
    reg, path, _ = open_registry(tmp_path)
    reg.append(make_record(family_id="F-anchor", variant="anchor"))
    before = path.read_bytes()

    if accepted:
        reg.append(make_record(family_id=family_id, variant="candidate"))
        assert [r.family_id for r in reg.records()] == ["F-anchor", family_id]
        return
    with pytest.raises((RecordRefused, TypeError) if family_id is None else RecordRefused):
        reg.append(make_record(family_id=family_id, variant="candidate"))
    assert path.read_bytes() == before


@pytest.mark.parametrize(
    ("status", "accepted"),
    [
        ("failed", True),
        ("scored", True),
        ("rejected", True),
        ("accepted", False),
        ("Adopted", False),
        ("SCORED", False),
        ("scored ", False),
        ("", False),
        ("pending", False),
    ],
)
def test_an_unknown_status_is_refused(tmp_path, status, accepted):
    reg, path, _ = open_registry(tmp_path)
    reg.append(make_record(variant="anchor"))
    before = path.read_bytes()

    if accepted:
        reg.append(make_record(variant="candidate", status=status))
        assert [r.status for r in reg.records()] == ["scored", status]
        return
    with pytest.raises(RecordRefused):
        reg.append(make_record(variant="candidate", status=status))
    assert path.read_bytes() == before


# --------------------------------------------------------------------------- family size


FAMILIES = {
    "FA": [("E1", "scored"), ("E1", "failed"), ("E1", "rejected")],
    "FB": [("L2", "failed")],
    "FC": [
        ("P3", "scored"),
        ("P3", "failed"),
        ("P4", "failed"),
        ("P4", "rejected"),
        ("P3", "scored"),
        ("P4", "adopted"),
    ],
}


def test_family_size_counts_failed_variants(tmp_path):
    reg, path, log = open_registry(tmp_path)
    assert reg.family_size("FA") == 0

    # Interleave the families so a count cannot lean on contiguous blocks.
    queues = {fam: list(enumerate(runs)) for fam, runs in FAMILIES.items()}
    while any(queues.values()):
        for fam, queue in queues.items():
            if not queue:
                continue
            i, (hid, status) = queue.pop(0)
            reg.append(
                make_record(
                    hypothesis_id=hid,
                    family_id=fam,
                    variant=f"v-{fam}-{i}",
                    status=status,
                    metrics=() if status == "failed" else (fold("wf-1", "ll", 0.7, 0.6, 0.8, 9),),
                    owner_signoff=141 if status == "adopted" else None,
                )
            )

    reopened = Registry(path, holdout_log=log)
    for registry in (reg, reopened):
        for fam, runs in FAMILIES.items():
            assert registry.family_size(fam) == len(runs), fam
            got = registry.records(fam)
            assert sorted(r.variant for r in got) == sorted(
                f"v-{fam}-{i}" for i in range(len(runs))
            )
            assert sorted(r.status for r in got) == sorted(status for _, status in runs)
        assert registry.family_size("FZ") == 0


def test_a_later_decision_on_a_variant_does_not_grow_the_family(tmp_path):
    """A variant is the run's specification (hypothesis_id, variant, fold_scheme, sources,
    components, params) plus the feature-table identity of each data build. A data_builds value
    ``"<identity>@<refresh>"`` splits at its last ``@``: the build name and the identity part
    count, the refresh id does not; a value with no ``@`` is identity in full. A decision, retry
    or rerun (new git_sha or ticket, or a refresh of the same builds) appended for a
    specification already tried is another run, not another variant; a change to params,
    components, fold_scheme, sources or a build's identity, or an added, removed or renamed
    build, is a new variant even under the same label. Mapping key order never matters."""
    reg, path, log = open_registry(tmp_path)
    base = make_record(
        hypothesis_id="E1",
        family_id="F1",
        variant="v1",
        components={"model": "logit-1.0", "calibration": "none"},
        params={"C": 1.0, "penalty": "l2"},
    )
    sweep = make_record(hypothesis_id="P3", family_id="FP", variant="logit")
    failed = {"status": "failed", "metrics": ()}
    adopted = {"status": "adopted", "owner_signoff": 141}
    new_build = {"results": hashlib.sha256(b"synthetic build 2").hexdigest()}

    def c(value: float) -> dict:
        return {"C": value, "penalty": "l2"}

    # FD compares feature tables under the one label "logit", all else equal. Identities and
    # refresh ids are both hex, so only the last "@" tells them apart.
    def hexid(seed: bytes) -> str:
        return hashlib.sha256(seed).hexdigest()

    RES, FEAT_A, FEAT_B, ODDS = (hexid(s) for s in (b"results", b"feat A", b"feat B", b"odds"))
    R1, R2 = hexid(b"rows to 2026-10-31")[:12], hexid(b"rows to 2026-11-24")[:12]

    def at(identity: str, refresh: str) -> str:
        return f"{identity}@{refresh}"

    def builds(features: str, results_refresh: str = R1, features_refresh: str = R1) -> dict:
        # Not in sorted key order, so an order-sensitive comparison has something to trip on.
        return {"results": at(RES, results_refresh), "features": at(features, features_refresh)}

    feat = make_record(
        hypothesis_id="P5", family_id="FD", variant="logit", data_builds=builds(FEAT_A)
    )

    # (family, spec, run, what it is) in append order, the families interleaved. ``spec`` names
    # the run's specification by hand: runs sharing a name are one variant, a new name is a new
    # variant. FP is a C sweep logged under the one label "logit": each C is its own variant.
    runs = [
        ("F1", "v1", base, "first run"),
        ("FD", "A", feat, "feature table A"),
        ("F2", "v1", replace(base, family_id="F2"), "the same spec in another family"),
        ("FP", "C=0.01", replace(sweep, params=c(0.01)), "sweep point"),
        ("FD", "B", replace(feat, data_builds=builds(FEAT_B)), "feature table B, all else equal"),
        ("F1", "v1", replace(base, status="rejected"), "rejected after comparison"),
        ("FP", "C=0.1", replace(sweep, params=c(0.1)), "sweep point: only params changed"),
        (
            "FD",
            "A",
            replace(feat, data_builds=builds(FEAT_A, R2, R2), ticket=75),
            "A refreshed: the same identities, newer rows",
        ),
        ("F1", "v2", replace(base, variant="v2", **failed), "a new label, failed"),
        ("F2", "v1", replace(base, family_id="F2", **adopted), "adopted later"),
        ("F1", "v2", replace(base, variant="v2"), "the retry, scored"),
        (
            "FD",
            "B",
            replace(feat, data_builds=builds(FEAT_B, results_refresh=R2), status="rejected"),
            "B with only its results build refreshed, rejected",
        ),
        ("FP", "C=1", replace(sweep, params=c(1.0)), "sweep point"),
        ("F1", "E2:v1", replace(base, hypothesis_id="E2"), "the same label, other hypothesis"),
        (
            "FD",
            "A",
            replace(
                feat,
                data_builds={"features": at(FEAT_A, R1), "results": at(RES, R1)},
                git_sha="f" * 40,
            ),
            "A rerun with the build names in another key order",
        ),
        ("F1", "v2", replace(base, variant="v2", **adopted), "adopted later"),
        ("F1", "v1", replace(base, git_sha="b" * 40), "rerun at a new git sha"),
        (
            "FD",
            "A +odds",
            replace(feat, data_builds={**builds(FEAT_A), "odds": at(ODDS, R1)}),
            "a build name added",
        ),
        ("FP", "C=10", replace(sweep, params=c(10.0)), "sweep point"),
        (
            "F1",
            "v1 build 2",
            replace(base, data_builds=new_build, ticket=75),
            "another bare build hash: a value without @ is identity in full",
        ),
        (
            "FD",
            "A -results",
            replace(feat, data_builds={"features": at(FEAT_A, R1)}),
            "a build name removed",
        ),
        (
            "F1",
            "v1",
            replace(
                base,
                git_sha="c" * 40,
                params={"penalty": "l2", "C": 1.0},
                components={"calibration": "none", "model": "logit-1.0"},
            ),
            "rerun with the mappings' keys in another order",
        ),
        ("FP", "C=100", replace(sweep, params=c(100.0)), "sweep point"),
        (
            "FD",
            "A renamed",
            replace(feat, data_builds={"results": at(RES, R1), "feature_table": at(FEAT_A, R1)}),
            "identity A under another build name",
        ),
        ("F1", "v1 C=2", replace(base, params=c(2.0)), "only params changed under v1"),
        ("F1", "v1 C=2", replace(base, params=c(2.0), status="rejected"), "a decision on it"),
        (
            "FD",
            "A@lag-7",
            replace(feat, data_builds=builds(f"{FEAT_A}@lag-7")),
            "an identity holding an @: only the last @ splits off the refresh id",
        ),
        (
            "F1",
            "v1 isotonic",
            replace(base, components={"model": "logit-1.0", "calibration": "isotonic-0.3"}),
            "only components changed under v1",
        ),
        ("FP", "C=1", replace(sweep, params=c(1.0), git_sha="d" * 40), "sweep point rerun"),
        (
            "FD",
            "A@lag-7",
            replace(feat, data_builds=builds(f"{FEAT_A}@lag-7", R2, R2), git_sha="9" * 40),
            "that identity refreshed",
        ),
        (
            "F1",
            "v1 wf-4",
            replace(base, fold_scheme="walk-forward-4", **failed),
            "only fold_scheme changed under v1, failed",
        ),
        ("F1", "v1 wf-4", replace(base, fold_scheme="walk-forward-4", git_sha="e" * 40), "retry"),
        (
            "F1",
            "v1 +jra",
            replace(base, sources=("netkeiba", "jra")),
            "only sources changed under v1",
        ),
        ("FP", "C=100", replace(sweep, params=c(100.0), status="rejected"), "a decision on it"),
        ("F1", "v3", replace(base, variant="v3", **failed), "failed, never retried"),
    ]

    tried: dict[str, set[str]] = {"F1": set(), "F2": set(), "FP": set(), "FD": set()}
    want_before = got_before = {fam: reg.family_size(fam) for fam in tried}
    assert got_before == dict.fromkeys(tried, 0)
    # Every row is checked against every family, and every miscounted row is reported: a row
    # miscounts when a family grows by other than what the spec names say.
    miscounted = []
    for i, (family, spec, run, what) in enumerate(runs):
        reg.append(run)
        tried[family].add(spec)
        want = {fam: len(specs) for fam, specs in tried.items()}
        got = {fam: reg.family_size(fam) for fam in tried}
        for fam in tried:
            grew, should = got[fam] - got_before[fam], want[fam] - want_before[fam]
            if grew != should:
                miscounted.append(
                    f"run {i} ({family}: {what}): family_size({fam!r}) grew by {grew}, not {should}"
                )
        want_before, got_before = want, got
    assert not miscounted, "\n".join(miscounted)
    assert want == {"F1": 9, "F2": 1, "FP": 5, "FD": 6}

    reopened = Registry(path, holdout_log=log)
    for registry in (reg, reopened):
        assert registry.family_size("F1") == 9
        assert registry.family_size("F2") == 1
        assert registry.family_size("FP") == 5
        assert registry.family_size("FD") == 6
        assert registry.family_size("F9") == 0
        # Every run is still there, decisions, retries and reruns included, in append order.
        assert registry.records() == [run for _, _, run, _ in runs]
        assert len(registry.records("FP")) == 7
        assert len(registry.records("FD")) == 10
        for fam in ("F1", "F2", "FP", "FD"):
            assert registry.records(fam) == [run for f, _, run, _ in runs if f == fam]
    reopened.verify()


# --------------------------------------------------------------------------- frozen families


SPECS = [
    {
        "metric": "log_loss",
        "variants": ["a", "b", "c"],
        "alpha": 5,
        "cluster": {"by": "meeting", "reps": 2000},
    },
    {"variants": ["only"], "flag": True, "none": None},
    {"z": 1, "a": [3, 2, 1], "m": {"y": "x", "b": "c"}},
]


def canonical_hash(spec: dict) -> str:
    text = json.dumps(spec, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def reordered(value):
    """The same JSON value with every mapping's keys in reverse order."""
    if isinstance(value, dict):
        return {key: reordered(value[key]) for key in reversed(list(value))}
    if isinstance(value, list):
        return [reordered(item) for item in value]
    return value


def test_a_frozen_family_cannot_be_refrozen_with_a_different_spec(tmp_path):
    assert issubclass(FamilyAlreadyFrozen, ValueError)
    reg, path, log = open_registry(tmp_path)
    reg.append(make_record(family_id="F-unfrozen", variant="a-run"))

    for i, spec in enumerate(SPECS):
        family = f"P-fam-{i}"
        assert reg.frozen_spec_hash(family) is None
        frozen = reg.freeze_family(family, spec)
        assert frozen == canonical_hash(spec)
        assert reg.frozen_spec_hash(family) == frozen
        assert reg.freeze_family(family, reordered(spec)) == frozen  # same spec, any key order
    assert len({canonical_hash(spec) for spec in SPECS}) == len(SPECS)

    changed_value = {**SPECS[0], "alpha": 6}
    extra_variant = {**SPECS[0], "variants": ["a", "b", "c", "d"]}
    extra_key = {**SPECS[1], "note": "added after the freeze"}
    list_order = {**SPECS[2], "a": [1, 2, 3]}
    for family, spec in (
        ("P-fam-0", changed_value),
        ("P-fam-0", extra_variant),
        ("P-fam-1", extra_key),
        ("P-fam-2", list_order),
    ):
        with pytest.raises(FamilyAlreadyFrozen):
            reg.freeze_family(family, spec)

    reopened = Registry(path, holdout_log=log)
    for i, spec in enumerate(SPECS):
        assert reopened.frozen_spec_hash(f"P-fam-{i}") == canonical_hash(spec)
    with pytest.raises(FamilyAlreadyFrozen):
        reopened.freeze_family("P-fam-1", SPECS[2])
    assert reopened.frozen_spec_hash("P-fam-1") == canonical_hash(SPECS[1])
    assert reopened.frozen_spec_hash("never-frozen") is None

    # A freeze is not a run.
    assert [r.variant for r in reopened.records()] == ["a-run"]
    assert reopened.family_size("F-unfrozen") == 1
    assert reopened.family_size("P-fam-0") == 0
    assert reopened.records("P-fam-0") == []
    reopened.verify()


# --------------------------------------------------------------------------- holdout link


def test_holdout_open_links_to_the_holdout_log(tmp_path):
    reg, path, log = open_registry(tmp_path)
    lines = append_holdout_opens(log, ["calibration check", "final evaluation", "re-check"])
    other_log = tmp_path / "countries" / "hongkong" / "holdout_log.jsonl"
    other_log.parent.mkdir(parents=True)
    other_lines = append_holdout_opens(other_log, ["hong kong check"], first_ticket=90)

    for i, line in enumerate(lines):
        reg.append(make_record(variant=f"opened-{i}", holdout_open=line_sha(line)))
    # The log is read as it is at append time, not when the registry was opened.
    (later,) = append_holdout_opens(log, ["a later open"], first_ticket=99)
    reg.append(make_record(variant="opened-later", holdout_open=line_sha(later)))
    expected = [line_sha(line) for line in [*lines, later]]
    assert [r.holdout_open for r in reg.records()] == expected

    before = path.read_bytes()
    refused = {
        "an unknown hash": hashlib.sha256(b"never logged").hexdigest(),
        "the hash of the line with its newline": line_sha(lines[0] + "\n"),
        "the hash of an empty line": hashlib.sha256(b"").hexdigest(),
        "a line of another country's log": line_sha(other_lines[0]),
        "a prefix of a real hash": line_sha(lines[0])[:16],
        "a real hash with a suffix": line_sha(lines[0]) + "00",
        "not a hash": "not-a-hash",
        "an empty string": "",
        "the log line itself": lines[0],
    }
    for what, value in refused.items():
        try:
            reg.append(make_record(variant="bad-open", holdout_open=value))
        except RecordRefused:
            continue
        pytest.fail(f"a holdout_open of {what} was accepted")
    assert path.read_bytes() == before

    # With no holdout log on disk, no open can match.
    bare_path = tmp_path / "data" / "registry" / "bare.jsonl"
    bare = Registry(bare_path, holdout_log=tmp_path / "missing" / "holdout_log.jsonl")
    with pytest.raises(RecordRefused):
        bare.append(make_record(variant="no-log", holdout_open=line_sha(lines[0])))
    assert file_bytes(bare_path) in (None, b"")
    bare.append(make_record(variant="no-open"))
    assert [r.holdout_open for r in bare.records()] == [None]


@pytest.mark.parametrize(
    ("signoff", "error"),
    [
        pytest.param(None, RecordRefused, id="none"),
        pytest.param(0, RecordRefused, id="zero"),
        pytest.param(-141, RecordRefused, id="negative"),
        pytest.param(True, (RecordRefused, TypeError), id="bool"),
        pytest.param("141", (RecordRefused, TypeError), id="string"),
        pytest.param(141.0, (RecordRefused, TypeError), id="float"),
    ],
)
def test_adopted_status_needs_an_owner_signoff_ticket(tmp_path, signoff, error):
    reg, path, _ = open_registry(tmp_path)
    reg.append(make_record(variant="scored-without-signoff"))
    before = path.read_bytes()

    with pytest.raises(error):
        reg.append(make_record(variant="adopted", status="adopted", owner_signoff=signoff))
    assert path.read_bytes() == before

    for ticket in (141, 9):
        reg.append(make_record(variant=f"adopted-{ticket}", status="adopted", owner_signoff=ticket))
    got = [(r.variant, r.status, r.owner_signoff) for r in reg.records()]
    assert got == [
        ("scored-without-signoff", "scored", None),
        ("adopted-141", "adopted", 141),
        ("adopted-9", "adopted", 9),
    ]


# --------------------------------------------------------------------------- committed summary


SHA_MARK = "5ca1ab1ec0ffee0ddba11c0de5eed5facade0042"
BUILD_MARK = "b0a710ad" * 8
SUMMARY_TICKET = 90817


def summary_records(holdout_hash: str) -> list[RunRecord]:
    common = {
        "ticket": SUMMARY_TICKET,
        "git_sha": SHA_MARK,
        "data_builds": {"buildkeyqx": BUILD_MARK},
        "sources": ("netkeiba",),
        "fold_scheme": "schemeqx-walk-6",
        "components": {"compkeyqx": "compverqx-3.1"},
    }
    return [
        make_record(
            **common,
            hypothesis_id="E31",
            family_id="FAM-alpha",
            variant="var-alpha-1",
            params={
                "paramkeyqx": "paramvalqx",
                "lrqx": 0.0371,
                "nestedqx": {"deepqx": "deepvalqx"},
            },
            metrics=(
                fold("foldqa-2021", "log_loss", 0.71, 0.63, 0.82, 977),
                fold("foldqa-2022", "brier", 0.58, 0.47, 0.66, 640),
            ),
            status="scored",
        ),
        make_record(
            **common,
            hypothesis_id="L8",
            family_id="FAM-beta",
            variant="var-beta-fail",
            params={"paramkeyqy": "paramvalqy"},
            metrics=(),
            status="failed",
        ),
        make_record(
            **common,
            hypothesis_id="P5",
            family_id="FAM-alpha",
            variant="var-alpha-2",
            params={"paramkeyqz": 4417},
            metrics=(
                fold("foldqa-2021", "roi", 0.39, 0.27, 0.44, 812),
                fold("foldqa-2022", "roi", 0.52, None, None, 503),
            ),
            status="rejected",
            holdout_open=holdout_hash,
        ),
    ]


def test_committed_summary_has_aggregates_only(tmp_path):
    reg, _, log = open_registry(tmp_path)
    (opened,) = append_holdout_opens(log, ["summary check"])
    holdout_hash = line_sha(opened)
    for record in summary_records(holdout_hash):
        reg.append(record)

    text = summary_of(reg, tmp_path)

    listed = [
        "FAM-alpha",
        "FAM-beta",
        "E31",
        "L8",
        "P5",
        "var-alpha-1",
        "var-beta-fail",
        "var-alpha-2",
        "scored",
        "failed",
        "rejected",
        "foldqa-2021",
        "foldqa-2022",
        "log_loss",
        "brier",
        "roi",
    ]
    numbers = ["0.71", "0.63", "0.82", "977", "0.58", "0.47", "0.66", "640"]
    numbers += ["0.39", "0.27", "0.44", "812", "0.52", "503"]
    missing = [item for item in listed if item not in text]
    missing += [num for num in numbers if not shows_number(text, num)]
    assert missing == [], f"the summary lacks {missing}"

    not_listed = [
        "paramkeyqx",
        "paramvalqx",
        "lrqx",
        "nestedqx",
        "deepqx",
        "deepvalqx",
        "paramkeyqy",
        "paramvalqy",
        "paramkeyqz",
        "buildkeyqx",
        BUILD_MARK,
        BUILD_MARK[:12],
        SHA_MARK,
        SHA_MARK[:7],
        "schemeqx",
        "compkeyqx",
        "compverqx",
        holdout_hash,
        holdout_hash[:12],
        "netkeiba",
    ]
    leaked = [item for item in not_listed if item in text]
    leaked += [num for num in ("0.0371", "4417", str(SUMMARY_TICKET)) if shows_number(text, num)]
    assert leaked == [], f"the summary carries fields that are not aggregates: {leaked}"


# The licensed run's numbers: every rounding to 2-6 decimals keeps the 2-decimal prefix.
LICENSED_METRICS = (
    fold("foldqb-2021", "log_loss", 0.123456, 0.08312, 0.16424, 4321),
    fold("foldqb-2022", "log_loss", 0.94321, 0.91234, 0.97111, 6789),
)
LICENSED_NUMBERS = ["0.123456", "0.12", "0.08", "0.16", "4321", "0.94", "0.91", "0.97", "6789"]
PUBLIC_METRICS = (fold("foldqb-2021", "log_loss", 0.71, 0.63, 0.82, 977),)
PUBLIC_NUMBERS = ["0.71", "0.63", "0.82", "977"]


def licensed_and_public(sources) -> tuple[RunRecord, RunRecord]:
    licensed = make_record(
        hypothesis_id="E77",
        family_id="FAM-lic",
        variant="var-licensed",
        sources=sources,
        metrics=LICENSED_METRICS,
    )
    public = make_record(
        hypothesis_id="E5",
        family_id="FAM-lic",
        variant="var-public",
        sources=("netkeiba",),
        metrics=PUBLIC_METRICS,
    )
    return licensed, public


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param(("netkeiba", "jrdb"), id="netkeiba-jrdb"),
        pytest.param(("jrdb",), id="jrdb"),
        pytest.param(("jravan",), id="jravan"),
        pytest.param(("jra", "jravan"), id="jra-jravan"),
        pytest.param(("jravan", "jrdb"), id="jravan-jrdb"),
    ],
)
def test_summary_withholds_metrics_of_licensed_source_runs(tmp_path, sources):
    reg, _, _ = open_registry(tmp_path)
    licensed, public = licensed_and_public(sources)
    reg.append(public)
    reg.append(licensed)
    assert reg.records() == [public, licensed]  # the registry itself keeps the values

    text = summary_of(reg, tmp_path)

    leaked = [num for num in LICENSED_NUMBERS if shows_number(text, num)]
    assert leaked == [], f"the summary shows values of a licensed-source run: {leaked}"
    assert WITHHELD_LABEL in text
    assert "E77" in text and "var-licensed" in text and "FAM-lic" in text
    assert "E5" in text and "var-public" in text
    shown = [num for num in PUBLIC_NUMBERS if shows_number(text, num)]
    assert shown == PUBLIC_NUMBERS, "a run with no licensed source must show its values"


@pytest.mark.parametrize(
    "sources",
    [
        pytest.param(("JRDB",), id="upper-case"),
        pytest.param(("netkeiba", "JraVan"), id="mixed-case"),
        pytest.param((" jrdb",), id="leading-space"),
        pytest.param("jrdb", id="bare-string"),
    ],
)
def test_a_licensed_source_in_another_spelling_is_still_withheld(tmp_path, sources):
    """Either the record is refused, or the summary withholds it; its values never show."""
    reg, path, _ = open_registry(tmp_path)
    _, public = licensed_and_public(("netkeiba",))
    reg.append(public)
    before = path.read_bytes()

    appended = False
    try:
        licensed, _ = licensed_and_public(sources)
        reg.append(licensed)
        appended = True
    except (RecordRefused, TypeError):
        assert path.read_bytes() == before

    text = summary_of(reg, tmp_path)
    leaked = [num for num in LICENSED_NUMBERS if shows_number(text, num)]
    assert leaked == [], f"the summary shows values of a licensed-source run: {leaked}"
    if appended:
        assert WITHHELD_LABEL in text
    assert [num for num in PUBLIC_NUMBERS if shows_number(text, num)] == PUBLIC_NUMBERS


# ------------------------------------------------------------------------- #431 licensed row order
#
# Contract (#431, D21): the registry is append-only and a study may append its licensed runs in the
# rank order of their results, so ``write_summary`` writes a family's licensed-source runs in an
# order that depends only on their identity, never on when they were appended or on their figures:
# the same runs in any append order, or carrying other figures, give a byte-identical family file.
# Every run of a repeated variant stays; among themselves they may keep their append order. The
# sort key is the implementation's. The index and the public family's file are unaffected.

ORDER_FAMILY = "FAM-lic-order"
ORDER_PUBLIC_FAMILY = "FAM-pub-order"
# Licensed variant -> its sources. The names' order is neither a figure order nor its reverse.
ORDER_VARIANTS = {
    "var-cedar": ("jrdb",),
    "var-alder": ("netkeiba", "jravan"),
    "var-fir": ("jrdb",),
    "var-birch": ("jravan",),
    "var-elm": ("jrdb", "jravan"),
    "var-dogwood": ("jrdb",),
}
ORDER_REPEATED = "var-birch"  # a failed run, then a scored run, of the same variant
ORDER_PUBLIC = {
    "pub-one": fold("foldqd-2025", "roi", 0.71, 0.63, 0.82, 977),
    "pub-two": fold("foldqd-2025", "roi", 0.58, 0.47, 0.66, 640),
}
# The licensed figures, lowest first; a run's rank picks its value, CI and n.
ORDER_VALUES = (-0.7341, -0.4127, -0.1883, 0.0936, 0.2571, 0.6184)
ORDER_NS = (1503, 2277, 3061, 3845, 4629, 5413)
ORDER_RANKS = {
    "first": {
        "var-cedar": 5,
        "var-alder": 1,
        "var-fir": 3,
        "var-birch": 0,
        "var-elm": 4,
        "var-dogwood": 2,
    },
    "second": {
        "var-cedar": 0,
        "var-alder": 4,
        "var-fir": 1,
        "var-birch": 5,
        "var-elm": 2,
        "var-dogwood": 3,
    },
}


def order_run(variant: str, status: str, ranks: dict[str, int]) -> RunRecord:
    if variant in ORDER_PUBLIC:
        return make_record(
            hypothesis_id="E432",
            family_id=ORDER_PUBLIC_FAMILY,
            variant=variant,
            metrics=(ORDER_PUBLIC[variant],),
        )
    rank = ranks[variant]
    value = ORDER_VALUES[rank]
    metrics = (fold("foldqd-2025", "roi", value, value - 0.0412, value + 0.0388, ORDER_NS[rank]),)
    return make_record(
        hypothesis_id="E431",
        family_id=ORDER_FAMILY,
        variant=variant,
        sources=ORDER_VARIANTS[variant],
        metrics=() if status == "failed" else metrics,
        status=status,
    )


def by_return(ranks: dict[str, int], *, best_first: bool) -> list[tuple[str, str]]:
    """The scored licensed runs in the order of their returns, as a study ranks them."""
    ranked = sorted(ORDER_VARIANTS, key=lambda v: ranks[v], reverse=best_first)
    return [(variant, "scored") for variant in ranked]


def test_summary_orders_licensed_runs_independently_of_append_order_and_figures(tmp_path):
    failed, scored = (ORDER_REPEATED, "failed"), (ORDER_REPEATED, "scored")
    pub_one, pub_two = ("pub-one", "scored"), ("pub-two", "scored")
    # The failed run right before its scored run, as a retry leaves them.
    worst_first = by_return(ORDER_RANKS["first"], best_first=False)
    worst_first.insert(worst_first.index(scored), failed)
    # name -> (figures, append order); the public runs keep their relative order throughout.
    arrangements = {
        "best-first": (
            ORDER_RANKS["first"],
            [pub_one, failed, *by_return(ORDER_RANKS["first"], best_first=True), pub_two],
        ),
        "worst-first": (ORDER_RANKS["first"], [*worst_first, pub_one, pub_two]),
        "shuffled": (
            ORDER_RANKS["first"],
            [
                ("var-fir", "scored"),
                pub_one,
                failed,
                ("var-dogwood", "scored"),
                ("var-cedar", "scored"),
                scored,
                pub_two,
                ("var-alder", "scored"),
                ("var-elm", "scored"),
            ],
        ),
        "other-figures-best-first": (
            ORDER_RANKS["second"],
            [failed, pub_one, pub_two, *by_return(ORDER_RANKS["second"], best_first=True)],
        ),
    }

    every_run = sorted(
        [failed, pub_one, pub_two, *by_return(ORDER_RANKS["first"], best_first=True)]
    )
    pages = {}
    for name, (ranks, order) in arrangements.items():
        assert sorted(order) == every_run, f"{name} does not append every run once"
        assert order.index(failed) < order.index(scored), name
        assert order.index(pub_one) < order.index(pub_two), name
        reg, _, _ = open_registry(tmp_path / name)
        for variant, status in order:
            reg.append(order_run(variant, status, ranks))
        out = summary_dir_in(tmp_path / name)
        write_summary(reg, out)
        pages[name] = written(out)

    family_file, public_file = f"{ORDER_FAMILY}.md", f"{ORDER_PUBLIC_FAMILY}.md"
    reference = pages["best-first"]
    assert set(reference) == {"README.md", family_file, public_file}
    assert all(set(files) == set(reference) for files in pages.values())
    index = reference["README.md"].decode("utf-8")
    assert f"({family_file})" in index and f"({public_file})" in index
    unchanged = ("README.md", public_file)
    changed = [(n, f) for n, files in pages.items() for f in unchanged if files[f] != reference[f]]
    assert changed == [], f"the index or the public family's file changed: {changed}"

    follows = [
        name for name, files in pages.items() if files[family_file] != reference[family_file]
    ]
    assert follows == [], (
        f"{family_file} differs from best-first's in {follows}: the order of its licensed rows "
        "follows the append order or the figures, so it publishes their rank (D21)"
    )

    text = reference[family_file].decode("utf-8")
    counts = {variant: text.count(variant) for variant in ORDER_VARIANTS}
    assert counts == {v: 2 if v == ORDER_REPEATED else 1 for v in ORDER_VARIANTS}, (
        "every run of every licensed variant must be listed, a repeated variant's too"
    )
    assert WITHHELD_LABEL in text
    figures = [f"{v:.4f}" for v in ORDER_VALUES] + [str(n) for n in ORDER_NS]
    assert [num for num in figures if shows_number(text, num)] == []


# --------------------------------------------------------------------------- default registry


def test_default_registry_lives_in_the_data_directory(tmp_path, monkeypatch):
    data = tmp_path / "data-dir"
    monkeypatch.setenv(DATA_DIR_VAR, str(data))
    records = {
        country: make_record(variant=f"default-{country}") for country in ("japan", "hongkong")
    }

    for country, record in records.items():
        default_registry(country).append(record)

    for country, record in records.items():
        path = data / "registry" / f"{country}.jsonl"
        assert path.is_file()
        assert Registry(path, holdout_log=tmp_path / "no-log.jsonl").records() == [record]
        assert default_registry(country).records() == [record]


# --------------------------------------------------------------------------- #525 family file names
#
# Contract (#525, owner's ruling on the ticket): ``append`` and ``freeze_family`` refuse a family id
# the summary cannot name: one that is not a safe file stem (an ASCII letter or digit, then letters,
# digits, ``.``, ``_`` or ``-``), or that is the index stem ``README`` in any case (macOS file names
# ignore case). ``append`` also refuses a new family whose summary file a different family already
# holds. Lines already in a registry are never edited: ``write_summary`` names a legacy family by
# mapping each ``/`` to ``--``, shows the original id in its file and in the index link, and raises
# ``ValueError`` before writing any file when a mapped name is still unsafe or two families would
# share one file (ignoring case).

UNNAMEABLE_FAMILIES = [
    "a/b",
    "a/b/c",
    "japan/131-exacta-category-v1",
    "japan/131-exacta-category-v1/max-test",
    "/x",
    "x/",
    "../x",
    "README",
    "readme",
    "Readme",
    ".hidden",
    "-lead",
    "_lead",
    "a b",
    "a\\b",
    "a:b",
    "F1\n",
    "ペース",
]
NAMEABLE_FAMILIES = [
    "F1",
    "0lead",
    "japan-131-exacta-category-v2",
    "japan-131-exacta-category-v1--max-test",
    "x.y_z-9",
    "READMEX",
    "README-v2",
]


def seal_lines(path: Path, entries: list[dict]) -> None:
    """Append ``entries`` as the registry writes its lines (canonical JSON linked by ``prev`` to the
    last line's sha256, sealed with the sha256 of the line without ``seal``), bypassing every check
    ``append`` and ``freeze_family`` make: the way records written before #525 sit in a real log."""

    def canonical(value) -> str:
        return json.dumps(
            value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        )

    data = path.read_bytes() if path.exists() else b""
    head = hashlib.sha256(data[:-1].split(b"\n")[-1]).hexdigest() if data else "0" * 64
    with open(path, "ab") as handle:
        for i, entry in enumerate(entries):
            body = {**entry, "prev": head, "utc": f"2026-11-2{i % 7}T08:00:00+00:00"}
            body["seal"] = hashlib.sha256(canonical(body).encode("ascii")).hexdigest()
            line = canonical(body).encode("ascii")
            handle.write(line + b"\n")
            head = hashlib.sha256(line).hexdigest()


def legacy_run(record: RunRecord) -> dict:
    data = asdict(record)
    text = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return {
        "kind": "run",
        "run_id": hashlib.sha256(text.encode("ascii")).hexdigest(),
        "record": data,
    }


def legacy_freeze(family_id: str, spec: dict) -> dict:
    return {
        "kind": "freeze",
        "family_id": family_id,
        "spec": spec,
        "spec_hash": canonical_hash(spec),
    }


def written(directory: Path) -> dict[str, bytes]:
    return (
        {p.name: p.read_bytes() for p in sorted(directory.iterdir())} if directory.exists() else {}
    )


def summary_dir_in(tmp_path: Path, name: str = "summary") -> Path:
    out = tmp_path / "docs" / "experiments" / "registry" / name
    out.mkdir(parents=True, exist_ok=True)
    return out


def test_append_refuses_a_family_id_the_summary_cannot_name(tmp_path):
    reg, path, _ = open_registry(tmp_path)
    reg.append(make_record(family_id="F-anchor", variant="anchor"))

    wrongly_accepted = []
    for i, family in enumerate(UNNAMEABLE_FAMILIES):
        before = path.read_bytes()
        try:
            reg.append(make_record(family_id=family, variant=f"bad-{i}"))
        except RecordRefused:
            assert path.read_bytes() == before, f"a refused {family!r} still wrote"
            continue
        wrongly_accepted.append(family)
    assert wrongly_accepted == [], (
        f"append accepted family ids the summary cannot name: {wrongly_accepted}"
    )
    reg.verify()
    assert [r.family_id for r in reg.records()] == ["F-anchor"]

    for i, family in enumerate(NAMEABLE_FAMILIES):
        reg.append(make_record(family_id=family, variant=f"good-{i}"))
    assert [r.family_id for r in reg.records()] == ["F-anchor", *NAMEABLE_FAMILIES]
    write_summary(reg, summary_dir_in(tmp_path))  # every accepted family can be named


def test_freeze_family_refuses_a_family_id_the_summary_cannot_name(tmp_path):
    reg, path, _ = open_registry(tmp_path)
    reg.freeze_family("P-anchor", SPECS[0])

    wrongly_frozen = []
    for family in UNNAMEABLE_FAMILIES:
        before = path.read_bytes()
        try:
            reg.freeze_family(family, SPECS[1])
        except RecordRefused:
            assert path.read_bytes() == before, f"a refused freeze of {family!r} still wrote"
            assert reg.frozen_spec_hash(family) is None
            continue
        wrongly_frozen.append(family)
    assert wrongly_frozen == [], f"freeze_family accepted unnameable family ids: {wrongly_frozen}"

    for i, family in enumerate(NAMEABLE_FAMILIES):
        assert reg.freeze_family(family, SPECS[i % len(SPECS)]) == canonical_hash(
            SPECS[i % len(SPECS)]
        )
    reg.verify()


# Legacy ids already in a log, and the summary file each must get.
LEGACY_FILES = {
    "japan/131-exacta-category-v1": "japan--131-exacta-category-v1.md",
    "japan/131-exacta-category-v1/max-test": "japan--131-exacta-category-v1--max-test.md",
    "a/b": "a--b.md",
    "X9/y.z_w": "X9--y.z_w.md",
}
NORMAL_FAMILIES = ["F-win", "japan-131-exacta-category-v2"]


def test_summary_names_legacy_slash_families_by_a_safe_file(tmp_path):
    reg, path, log = open_registry(tmp_path)
    normal = [
        make_record(
            family_id=family,
            variant=f"var-normal-{i}",
            metrics=(fold("wf-1", "roi", 0.4 + i, 0.1, 0.9, 30 + i),),
        )
        for i, family in enumerate(NORMAL_FAMILIES)
    ]
    reg.append(normal[0])
    legacy = {
        family: [
            make_record(
                family_id=family, variant=f"var-legacy-{i}-{j}", hypothesis_id=f"E{131 + i}"
            )
            for j in range(2)
        ]
        for i, family in enumerate(LEGACY_FILES)
    }
    entries = [legacy_freeze("japan/131-exacta-category-v1", SPECS[2])]
    for j in range(2):  # interleaved, as parallel runs would leave them
        entries += [legacy_run(records[j]) for records in legacy.values()]
    seal_lines(path, entries)
    reg.append(normal[1])  # the chain goes on after the legacy lines

    reg.verify()
    assert reg.frozen_spec_hash("japan/131-exacta-category-v1") == canonical_hash(SPECS[2])
    for family, records in legacy.items():
        assert reg.records(family) == records

    out = summary_dir_in(tmp_path)
    write_summary(reg, out)
    expected = {"README.md", *LEGACY_FILES.values(), *(f"{f}.md" for f in NORMAL_FAMILIES)}
    assert set(written(out)) == expected

    index = (out / "README.md").read_text(encoding="utf-8")
    all_variants = {r.variant for records in legacy.values() for r in records} | {
        r.variant for r in normal
    }
    for family, name in LEGACY_FILES.items():
        text = (out / name).read_text(encoding="utf-8")
        assert family in text, f"{name} does not show the original id {family!r}"
        own = {r.variant for r in legacy[family]}
        assert {v for v in all_variants if v in text} == own
        assert re.search(r"\[" + re.escape(family) + r"\]\(" + re.escape(name) + r"\)", index), (
            f"the index does not link {family!r} to {name}"
        )

    # A normal family's file is the same as when no legacy family is in the registry.
    alone, _, _ = open_registry(tmp_path / "alone")
    for record in normal:
        alone.append(record)
    alone_out = summary_dir_in(tmp_path / "alone")
    write_summary(alone, alone_out)
    for family in NORMAL_FAMILIES:
        assert (out / f"{family}.md").read_bytes() == (alone_out / f"{family}.md").read_bytes()


# (families whose lines are in the log before the summary is first written, the family that then
# arrives and would share a file with one of them)
SHARED_FILES = [
    pytest.param(["a/b"], "a--b", id="slash-vs-dashes"),
    pytest.param(["japan/131-x"], "JAPAN--131-X", id="slash-vs-dashes-other-case"),
    pytest.param(["japan/ok", "a--b"], "a/b", id="dashes-then-slash"),
    pytest.param(["japan/ok", "FAM-Case"], "fam-case", id="case-only"),
    pytest.param(["japan/ok", "x.y"], "X.Y", id="case-only-dots"),
]


@pytest.mark.parametrize(("first", "colliding"), SHARED_FILES)
def test_summary_refuses_two_families_that_share_a_file(tmp_path, first, colliding):
    reg, path, _ = open_registry(tmp_path)
    seal_lines(
        path, [legacy_run(make_record(family_id=f, variant=f"v-{i}")) for i, f in enumerate(first)]
    )
    out = summary_dir_in(tmp_path)
    write_summary(reg, out)
    before = written(out)
    assert len(before) == len(first) + 1

    seal_lines(path, [legacy_run(make_record(family_id=colliding, variant="v-colliding"))])
    reg.verify()
    with pytest.raises(ValueError):
        write_summary(reg, out)
    assert written(out) == before, "a refused summary still changed the directory"


@pytest.mark.parametrize(
    "unsafe",
    ["/x", "../x", "README", "readme", ".hidden/x", "-/x", "a b/c", "x/ペース"],
)
def test_summary_refuses_a_legacy_family_with_no_safe_file_name(tmp_path, unsafe):
    reg, path, _ = open_registry(tmp_path)
    seal_lines(path, [legacy_run(make_record(family_id="japan/ok", variant="v-ok"))])
    reg.append(make_record(family_id="F-ok", variant="v-normal"))
    out = summary_dir_in(tmp_path)
    write_summary(reg, out)
    before = written(out)
    assert set(before) == {"README.md", "japan--ok.md", "F-ok.md"}

    seal_lines(path, [legacy_run(make_record(family_id=unsafe, variant="v-unsafe"))])
    reg.verify()
    with pytest.raises(ValueError):
        write_summary(reg, out)
    assert written(out) == before, "a refused summary still changed the directory"
    assert sorted(p.name for p in tmp_path.rglob("*.md")) == sorted(before), (
        "wrote outside the summary"
    )


def test_append_refuses_a_family_whose_summary_file_is_taken(tmp_path):
    reg, path, _ = open_registry(tmp_path)
    seal_lines(path, [legacy_run(make_record(family_id="a/b", variant="v-legacy"))])
    reg.append(make_record(family_id="FAM-Case", variant="v-first"))

    wrongly_accepted = []
    for i, family in enumerate(["a--b", "A--B", "fam-case", "FAM-CASE", "Fam-Case"]):
        before = path.read_bytes()
        try:
            reg.append(make_record(family_id=family, variant=f"v-taken-{i}"))
        except RecordRefused:
            assert path.read_bytes() == before
            continue
        wrongly_accepted.append(family)
    assert wrongly_accepted == [], (
        f"append accepted families whose file is taken: {wrongly_accepted}"
    )

    reg.append(make_record(family_id="FAM-Case", variant="v-second"))  # the same family is fine
    reg.append(make_record(family_id="a-b", variant="v-other"))  # a different file is fine
    assert [r.family_id for r in reg.records()] == ["a/b", "FAM-Case", "FAM-Case", "a-b"]
    out = summary_dir_in(tmp_path)
    write_summary(reg, out)
    assert set(written(out)) == {"README.md", "a--b.md", "FAM-Case.md", "a-b.md"}
