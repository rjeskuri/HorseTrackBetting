"""CI parallelism (ticket #600): the verify job stays well under its timeout.

The suite runs in parallel with pytest-xdist from scripts/verify.sh, the dependency is declared and
locked, and the verify job's timeout is at most 30 minutes (#600 set 20; the owner raised it to 30
in #697 while many PRs are in flight). These tests read the config files; they never run the suite.
"""

import re
import shlex
from pathlib import Path

import yaml

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
MAX_TIMEOUT_MINUTES = 30


def _pytest_commands() -> list[list[str]]:
    """Each non-comment line of verify.sh that runs pytest, as shell tokens."""
    commands = []
    for line in (ROOT / "scripts" / "verify.sh").read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        tokens = shlex.split(stripped, comments=True)
        if "pytest" in tokens:
            commands.append(tokens)
    return commands


def _worker_count(tokens: list[str]) -> str | None:
    """The value given to -n / --numprocesses in `tokens`, or None when absent."""
    for i, token in enumerate(tokens):
        if token in ("-n", "--numprocesses"):
            return tokens[i + 1] if i + 1 < len(tokens) else ""
        if token.startswith("--numprocesses="):
            return token.split("=", 1)[1]
        if re.fullmatch(r"-n.+", token):
            return token[2:]
    return None


def _name(requirement: str) -> str:
    return re.split(r"[\s<>=!~;\[]", requirement.strip(), maxsplit=1)[0].lower().replace("_", "-")


def test_verify_sh_runs_pytest_with_xdist_workers():
    commands = _pytest_commands()
    assert commands, "scripts/verify.sh has no line that runs pytest"
    for tokens in commands:
        workers = _worker_count(tokens)
        line = " ".join(tokens)
        assert workers is not None, f"pytest line has no xdist worker count (-n N): {line!r}"
        assert workers in ("auto", "logical") or (workers.isdigit() and int(workers) >= 1), (
            f"xdist worker count must be a positive integer, 'auto' or 'logical', got {workers!r}"
            f" in {line!r}"
        )


def test_dev_dependency_group_lists_pytest_xdist():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text())
    dev = data["dependency-groups"]["dev"]
    names = [_name(r) for r in dev if isinstance(r, str)]
    assert "pytest-xdist" in names, f"pytest-xdist is missing from [dependency-groups] dev: {dev}"


def test_ci_verify_job_timeout_is_at_most_30_minutes():
    workflow = yaml.safe_load((ROOT / ".github" / "workflows" / "ci.yml").read_text())
    job = workflow["jobs"]["verify"]
    assert "timeout-minutes" in job, "the verify job has no timeout-minutes"
    timeout = job["timeout-minutes"]
    assert isinstance(timeout, int) and not isinstance(timeout, bool), (
        f"timeout-minutes must be a plain number, got {timeout!r}"
    )
    assert 0 < timeout <= MAX_TIMEOUT_MINUTES, (
        f"verify job timeout-minutes is {timeout}, must be at most {MAX_TIMEOUT_MINUTES}"
    )


def test_uv_lock_has_a_pytest_xdist_package_entry():
    lock = tomllib.loads((ROOT / "uv.lock").read_text())
    names = [p.get("name", "").lower().replace("_", "-") for p in lock.get("package", [])]
    assert "pytest-xdist" in names, (
        "uv.lock has no pytest-xdist package entry, so `uv sync --locked` would not install it"
    )
    project_name = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]["name"]
    project = next(
        p
        for p in lock["package"]
        if p["name"].lower().replace("_", "-") == project_name.lower().replace("_", "-")
    )
    locked_dev = [d["name"].lower().replace("_", "-") for d in project["dev-dependencies"]["dev"]]
    assert "pytest-xdist" in locked_dev, (
        f"uv.lock's dev group for the project does not list pytest-xdist: {locked_dev}"
    )
