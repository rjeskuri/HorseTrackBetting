"""Ticket #334: the shared append-only docs merge without conflicts.

Contract pinned here (the coder implements what these tests call):

- ``write_summary(registry, directory)`` writes one file per family, ``<family_id>.md``, holding
  the writer's title and "do not edit by hand" preamble and exactly one ``## <family_id>`` table
  section, plus ``README.md``, the index: one ``- [<family>](<family>.md)`` list line per family
  file, in family-id order. A family's file does not change when another family is added.
- ``summary_dir(country)`` is ``<repo>/docs/experiments/registry/<country>``.
- ``.gitattributes`` marks ``docs/ledger.md``, ``docs/experiments/registry/**`` and the study
  reports ``docs/experiments/*.md`` with ``merge=union``, so two branches that each append to
  them merge with no conflict.
- ``docs/experiments/registry-japan.md`` is gone, split into the per-family files.
- CLAUDE.md "Where things live" states the append-only rule.

Merge tests run real ``git`` in a tmp repo. Every value is synthetic.
"""

from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

import pytest

from horsetrack.core.eval.registry import FoldMetric, Registry, RunRecord, write_summary

AT = "@"  # assembled so the public-tree scan sees no literal email
REPO_ROOT = Path(__file__).resolve().parents[1]
GIT_SHA = "0f1e2d3c4b5a69788796a5b4c3d2e1f00f1e2d3c"
PREAMBLE_TITLE = "# Experiment registry summary"
REGISTRY_REL = Path("docs/experiments/registry/japan")
LEDGER_REL = Path("docs/ledger.md")
REPORT_REL = Path("docs/experiments/study-japan.md")
OLD_SUMMARY = REPO_ROOT / "docs" / "experiments" / "registry-japan.md"
# Families of the old single-file summary; the committed directory keeps one file for each.
OLD_FAMILIES = {
    "japan-v0-win-residual",
    "japan-crowd-calibration-final-win",
    "japan-tail-beta-final-win",
    "japan-coplacer-place-e15",
    "japan-c4-meta-final-win",
    "japan-track-bias-day-sd",
    "japan-calibrated-market-final-win",
    "win5-carryover-v1",
}


# --------------------------------------------------------------------------- helpers


def run_record(family: str, variant: str, value: float, *, n: int = 100) -> RunRecord:
    return RunRecord(
        hypothesis_id="E1",
        family_id=family,
        variant=variant,
        ticket=334,
        git_sha=GIT_SHA,
        data_builds={"results": "b" * 64},
        sources=("netkeiba",),
        fold_scheme="walk-forward-3",
        components={"model": "logit"},
        params={"C": value},
        metrics=(FoldMetric(fold="all", metric="log_loss", value=value, lo=None, hi=None, n=n),),
        status="scored",
    )


def render(tmp_path: Path, name: str, records: list[RunRecord], dest: Path) -> None:
    """Write the summary of a fresh registry holding ``records`` into ``dest``."""
    root = tmp_path / "registries" / name
    root.mkdir(parents=True, exist_ok=True)
    registry = Registry(root / "japan.jsonl", holdout_log=root / "holdout_log.jsonl")
    for record in records:
        registry.append(record)
    dest.mkdir(parents=True, exist_ok=True)
    write_summary(registry, dest)


def family_files(directory: Path) -> dict[str, Path]:
    return {p.stem: p for p in directory.glob("*.md") if p.name != "README.md"}


def index_links(directory: Path) -> list[str]:
    """Family names the index links, in the order of its list lines."""
    text = (directory / "README.md").read_text(encoding="utf-8")
    links = []
    for line in text.splitlines():
        if line.lstrip().startswith(("-", "*")):
            match = re.search(r"\]\(([^)]+)\)", line)
            assert match, f"index list line without a link: {line!r}"
            assert match.group(1).endswith(".md"), line
            links.append(match.group(1)[: -len(".md")])
    return links


def registry_dir_problems(directory: Path) -> list[str]:
    """What is wrong with a registry directory: the index/file set check and the file shape."""
    problems = []
    files = family_files(directory)
    links = index_links(directory)
    if len(links) != len(set(links)):
        problems.append(f"index links a family twice: {links}")
    if set(links) != set(files):
        problems.append(f"index links {sorted(set(links))} but files are {sorted(files)}")
    for stem, path in files.items():
        lines = path.read_text(encoding="utf-8").splitlines()
        if not lines or lines[0] != PREAMBLE_TITLE:
            problems.append(f"{path.name} does not start with the writer's title")
        if not any("do not edit by hand" in line for line in lines[:8]):
            problems.append(f"{path.name} lacks the do-not-edit preamble")
        sections = [line for line in lines if line.startswith("## ")]
        if sections != [f"## {stem}"]:
            problems.append(f"{path.name} sections are {sections}")
    return problems


def cells(line: str) -> list[str]:
    return re.split(r"(?<!\\)\|", line.strip())[1:-1]


def table_rows(text: str) -> tuple[list[str], list[str]]:
    """(data rows, problems) of every pipe table in ``text``; a row must have its header's
    cell count."""
    rows, problems, block = [], [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        if block:
            width = len(cells(block[0]))
            for row in block[2:]:
                rows.append(row)
                if len(cells(row)) != width:
                    problems.append(f"{len(cells(row))} cells, header has {width}: {row[:80]}")
            block = []
    return rows, problems


def duplicate_claims(text: str) -> list[str]:
    """Claim cells (first column) that appear in more than one row of the same pipe table.
    A union merge of two amendments to one row keeps both versions; this finds that."""
    duplicates, block = [], []
    for line in text.splitlines() + [""]:
        if line.startswith("|"):
            block.append(line)
            continue
        claims = [cells(row)[0].strip() for row in block[2:]]
        duplicates += sorted({claim for claim in claims if claims.count(claim) > 1})
        block = []
    return duplicates


def git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_CONFIG_GLOBAL="/dev/null", GIT_CONFIG_SYSTEM="/dev/null")
    result = subprocess.run(
        ["git", *args], cwd=repo, env=env, capture_output=True, text=True, check=False
    )
    if check and result.returncode != 0:
        pytest.fail(f"git {' '.join(args)} failed: {result.stdout}{result.stderr}")
    return result


LEDGER_SEED = """# Known-vs-assumed ledger

## Japan results

| Claim | Label | Evidence | Retest by |
|---|---|---|---|
| Seed claim one. | measured | report a | none |
| Seed claim two. | single-run | report b | rerun |

## Japan rules

| Claim | Label | Evidence | Retest by |
|---|---|---|---|
| Rules claim. | assumption | none | test it |
"""

REPORT_SEED = """# Study report

## Seed section

Seed text.
"""


def ledger_row(tag: str) -> str:
    return f"| Claim {tag}. | measured | report {tag} | rerun {tag} |"


BASE_RECORDS = [run_record("fam-base", "base-v1", 0.5)]
BRANCH_RECORDS = {
    "a": [run_record("fam-base", "a-variant", 0.61, n=111), run_record("fam-a-new", "a-new", 0.7)],
    "b": [run_record("fam-base", "b-variant", 0.62, n=222), run_record("fam-b-new", "b-new", 0.8)],
}


def build_and_merge(
    tmp_path: Path, *, attributes: bool
) -> tuple[Path, subprocess.CompletedProcess]:
    """A tmp repo: a base commit, branches a and b that each append a ledger row, a registry
    family file (the real writer), a row to an existing family file and a study-report section.
    Returns the repo and the result of merging b into a."""
    attributes_file = REPO_ROOT / ".gitattributes"
    assert attributes_file.is_file(), "the repo has no .gitattributes"
    repo = tmp_path / "repo"
    repo.mkdir()
    git(repo, "init", "-q")
    git(repo, "config", "user.name", "Test")
    git(repo, "config", "user.email", "test" + AT + "example.invalid")
    git(repo, "config", "commit.gpgsign", "false")
    git(repo, "config", "core.hooksPath", "/dev/null")
    git(repo, "checkout", "-q", "-b", "main")
    if attributes:
        (repo / ".gitattributes").write_text(attributes_file.read_text(encoding="utf-8"))
    for rel in (LEDGER_REL, REPORT_REL):
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
    (repo / LEDGER_REL).write_text(LEDGER_SEED, encoding="utf-8")
    (repo / REPORT_REL).write_text(REPORT_SEED, encoding="utf-8")
    render(tmp_path, "base", BASE_RECORDS, repo / REGISTRY_REL)
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "base")

    for tag in ("a", "b"):
        git(repo, "checkout", "-q", "-b", tag, "main")
        ledger = (repo / LEDGER_REL).read_text(encoding="utf-8")
        # appended after the last row of the first table, and at the end of the file
        first_end = ledger.index("\n\n## Japan rules")
        ledger = ledger[:first_end] + "\n" + ledger_row(f"{tag}1") + ledger[first_end:]
        ledger += ledger_row(f"{tag}2") + "\n"
        (repo / LEDGER_REL).write_text(ledger, encoding="utf-8")
        report = (repo / REPORT_REL).read_text(encoding="utf-8")
        report += f"\n## Section {tag}\n\nText of section {tag}.\n"
        (repo / REPORT_REL).write_text(report, encoding="utf-8")
        render(tmp_path, tag, BASE_RECORDS + BRANCH_RECORDS[tag], repo / REGISTRY_REL)
        git(repo, "add", "-A")
        git(repo, "commit", "-q", "-m", f"branch {tag}")

    git(repo, "checkout", "-q", "a")
    return repo, git(repo, "merge", "--no-edit", "b", check=False)


# --------------------------------------------------------------------------- the writer


def test_write_summary_writes_one_file_per_family_and_an_index(tmp_path):
    records = [
        run_record("fam-zeta", "z1", 0.31),
        run_record("fam-alpha", "a1", 0.42),
        run_record("fam-mid", "m1", 0.53),
        run_record("fam-alpha", "a2", 0.64, n=222),
    ]
    out = tmp_path / "out"
    render(tmp_path, "w", records, out)

    assert set(family_files(out)) == {"fam-zeta", "fam-alpha", "fam-mid"}
    assert index_links(out) == ["fam-alpha", "fam-mid", "fam-zeta"]
    assert registry_dir_problems(out) == []
    alpha = family_files(out)["fam-alpha"].read_text(encoding="utf-8")
    assert "a1" in alpha and "a2" in alpha and "0.42" in alpha and "0.64" in alpha
    assert "z1" not in alpha and "m1" not in alpha
    zeta = family_files(out)["fam-zeta"].read_text(encoding="utf-8")
    assert "z1" in zeta and "a1" not in zeta


def test_a_family_file_does_not_change_when_another_family_is_added(tmp_path):
    first, second, third = tmp_path / "one", tmp_path / "two", tmp_path / "three"
    base = [run_record("fam-keep", "k1", 0.41), run_record("fam-keep", "k2", 0.52, n=7)]
    render(tmp_path, "one", base, first)
    render(tmp_path, "two", base + [run_record("fam-other", "o1", 0.9)], second)
    render(tmp_path, "three", base, third)

    keep = "fam-keep.md"
    assert (first / keep).read_bytes() == (second / keep).read_bytes()
    assert (first / keep).read_bytes() == (third / keep).read_bytes()  # a rerun is byte-identical
    assert (first / "README.md").read_bytes() == (third / "README.md").read_bytes()
    assert (first / "README.md").read_bytes() != (second / "README.md").read_bytes()


def test_summary_dir_is_the_country_folder_under_docs_experiments_registry():
    from horsetrack.core.eval.registry import summary_dir

    for country in ("japan", "hongkong"):
        assert summary_dir(country) == REPO_ROOT / "docs" / "experiments" / "registry" / country


# --------------------------------------------------------------------------- the merge


def test_two_branches_appending_to_the_shared_docs_merge_cleanly(tmp_path):
    repo, merge = build_and_merge(tmp_path, attributes=True)

    assert merge.returncode == 0, merge.stdout + merge.stderr
    assert git(repo, "diff", "--name-only", "--diff-filter=U").stdout == ""
    for path in repo.rglob("*"):
        if path.is_file() and ".git" not in path.parts:
            text = path.read_text(encoding="utf-8")
            assert "<<<<<<<" not in text and ">>>>>>>" not in text, path

    ledger = (repo / LEDGER_REL).read_text(encoding="utf-8")
    for tag in ("a1", "a2", "b1", "b2"):
        assert ledger.count(ledger_row(tag)) == 1, tag
    assert "| Seed claim one. |" in ledger and "| Rules claim. |" in ledger
    rows, problems = table_rows(ledger)
    assert problems == []
    assert len(rows) == len(set(rows)), "duplicate ledger rows after the merge"
    assert len(rows) == 7  # 3 seed rows and the 4 appended

    merged = repo / REGISTRY_REL
    assert registry_dir_problems(merged) == []
    assert set(family_files(merged)) == {"fam-base", "fam-a-new", "fam-b-new"}
    base_text = (merged / "fam-base.md").read_text(encoding="utf-8")
    for variant in ("base-v1", "a-variant", "b-variant"):
        assert base_text.count(variant) == 1, variant
    base_rows, base_problems = table_rows(base_text)
    assert base_problems == [] and len(base_rows) == 3
    assert "a-new" in (merged / "fam-a-new.md").read_text(encoding="utf-8")
    assert "b-new" in (merged / "fam-b-new.md").read_text(encoding="utf-8")

    report = (repo / REPORT_REL).read_text(encoding="utf-8")
    for heading in ("## Seed section", "## Section a", "## Section b"):
        assert report.count(heading) == 1, heading


def test_without_gitattributes_the_same_branches_conflict(tmp_path):
    repo, merge = build_and_merge(tmp_path, attributes=False)

    assert merge.returncode != 0
    conflicted = set(git(repo, "diff", "--name-only", "--diff-filter=U").stdout.split())
    expected = {
        str(LEDGER_REL),
        str(REPORT_REL),
        str(REGISTRY_REL / "fam-base.md"),
        str(REGISTRY_REL / "README.md"),
    }
    assert expected <= conflicted


# --------------------------------------------------------------------------- the committed repo


@pytest.mark.parametrize(
    "path",
    [
        "docs/ledger.md",
        "docs/experiments/registry/japan/japan-v0-win-residual.md",
        "docs/experiments/registry/japan/README.md",
        "docs/experiments/registry/hongkong/some-family.md",
        "docs/experiments/crowd-calibration-japan.md",
        "docs/experiments/coplacer-japan.md",
    ],
)
def test_gitattributes_marks_the_shared_docs_for_union_merge(path):
    result = git(REPO_ROOT, "check-attr", "merge", "--", path)
    assert result.stdout.strip() == f"{path}: merge: union"


def test_gitattributes_does_not_union_merge_code():
    result = git(REPO_ROOT, "check-attr", "merge", "--", "src/horsetrack/core/eval/registry.py")
    assert result.stdout.strip().endswith("merge: unspecified")


def test_the_committed_ledger_has_no_duplicate_rows_and_parseable_tables():
    text = (REPO_ROOT / LEDGER_REL).read_text(encoding="utf-8")
    rows, problems = table_rows(text)
    assert len(rows) > 20  # the ledger's tables were found
    assert problems == []
    duplicates = sorted({row for row in rows if rows.count(row) > 1})
    assert duplicates == []


def test_the_committed_registry_directory_matches_its_index():
    directory = REPO_ROOT / REGISTRY_REL
    assert (directory / "README.md").is_file(), "no registry index"
    assert registry_dir_problems(directory) == []
    assert OLD_FAMILIES <= set(family_files(directory))
    for stem, path in family_files(directory).items():
        rows, problems = table_rows(path.read_text(encoding="utf-8"))
        assert rows and problems == [], stem


def test_the_committed_ledger_claims_are_unique_within_each_table():
    text = (REPO_ROOT / LEDGER_REL).read_text(encoding="utf-8")
    assert duplicate_claims(text) == []


def test_claim_check_finds_the_duplicate_a_union_merge_of_two_amendments_leaves(tmp_path):
    repo, merge = build_and_merge(tmp_path, attributes=True)
    assert merge.returncode == 0, merge.stdout + merge.stderr
    merged = (repo / LEDGER_REL).read_text(encoding="utf-8")
    assert duplicate_claims(merged) == []  # appending rows is not a duplicate

    # Two branches amend the label cell of the same existing row, differently.
    original = "| Seed claim one. | measured | report a | none |"
    for tag in ("x", "y"):
        git(repo, "checkout", "-q", "-b", f"amend-{tag}", "main")
        text = (repo / LEDGER_REL).read_text(encoding="utf-8")
        assert original in text
        amended = f"| Seed claim one. | single-run {tag} | report a | none |"
        (repo / LEDGER_REL).write_text(text.replace(original, amended), encoding="utf-8")
        git(repo, "commit", "-q", "-am", f"amend {tag}")
    git(repo, "checkout", "-q", "amend-x")
    result = git(repo, "merge", "--no-edit", "amend-y", check=False)

    assert result.returncode == 0, result.stdout + result.stderr  # union: no conflict, silently
    text = (repo / LEDGER_REL).read_text(encoding="utf-8")
    assert "single-run x" in text and "single-run y" in text
    assert duplicate_claims(text) == ["Seed claim one."]
    assert "| Rules claim. |" in text  # only the amended row is reported


def test_claim_check_scopes_uniqueness_to_one_table():
    text = (
        "| Claim | Label |\n|---|---|\n| Same. | measured |\n| Other. | measured |\n\n"
        "| Claim | Label |\n|---|---|\n| Same. | assumption |\n"
    )
    assert duplicate_claims(text) == []
    assert duplicate_claims(text + "| Same. | single-run |\n") == ["Same."]
    assert duplicate_claims(text.replace("| Other. |", "| Same. |", 1)) == ["Same."]


def test_the_committed_family_files_start_as_the_writer_writes_them(tmp_path):
    out = tmp_path / "out"
    families = sorted(family_files(REPO_ROOT / REGISTRY_REL))
    assert families
    # a fresh family rendered through the real writer; the preamble does not depend on the family
    render(tmp_path, "pre", [run_record("fam-pre", "p1", 0.4)], out)
    written = (out / "fam-pre.md").read_text(encoding="utf-8")
    expected = written[: written.index("\n## ") + 1]
    assert expected.startswith(PREAMBLE_TITLE)
    for family in families:
        committed = (REPO_ROOT / REGISTRY_REL / f"{family}.md").read_text(encoding="utf-8")
        assert committed[: committed.index("\n## ") + 1] == expected, family


def test_the_single_file_registry_summary_is_gone():
    assert not OLD_SUMMARY.exists()


def test_claude_md_states_the_append_only_rule():
    text = (REPO_ROOT / "CLAUDE.md").read_text(encoding="utf-8")
    start = text.index("## Where things live")
    end = text.find("\n## ", start + 1)
    section = text[start : end if end != -1 else len(text)].lower()
    assert "append-only" in section
    assert "union" in section
