"""The horsetrack CLI root: the `japan` group, `--version`, and registration of later commands."""

import os
import re
import subprocess
import sys
from pathlib import Path

import pytest
import typer
from typer.testing import CliRunner

import horsetrack

ROOT = Path(__file__).resolve().parent.parent
runner = CliRunner()


def _app_and_japan():
    from horsetrack.cli import app, japan

    return app, japan


@pytest.fixture
def japan_group():
    """The japan sub-app, with any command a test registers removed afterwards."""
    _, japan = _app_and_japan()
    before = list(japan.registered_commands)
    yield japan
    japan.registered_commands[:] = before


def test_root_help_lists_the_japan_group():
    app, _ = _app_and_japan()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "japan" in result.output
    assert "--version" in result.output


def test_japan_group_help_exits_zero():
    app, _ = _app_and_japan()
    result = runner.invoke(app, ["japan", "--help"])
    assert result.exit_code == 0
    assert "Usage" in result.output


def test_version_prints_the_package_version():
    app, _ = _app_and_japan()
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert horsetrack.__version__ in result.output
    assert horsetrack.__version__ in result.output.strip().split()


def test_app_and_japan_are_typer_apps_with_japan_mounted():
    app, japan = _app_and_japan()
    assert isinstance(app, typer.Typer)
    assert isinstance(japan, typer.Typer)
    assert japan is not app
    assert [g.name for g in app.registered_groups] == ["japan"]
    assert app.registered_groups[0].typer_instance is japan


def test_bare_invocations_show_help_instead_of_running_anything():
    app, _ = _app_and_japan()
    root = runner.invoke(app, [])
    assert "japan" in root.output
    assert "Usage" in root.output
    japan = runner.invoke(app, ["japan"])
    assert "Usage" in japan.output
    assert "horsetrack japan" in japan.output or "japan" in japan.output


def test_unknown_command_fails_nonzero():
    app, _ = _app_and_japan()
    assert runner.invoke(app, ["nonesuch"]).exit_code != 0
    assert runner.invoke(app, ["japan", "nonesuch"]).exit_code != 0


@pytest.mark.parametrize("name,arg", [("ping", "abc"), ("echo-back", "x 1"), ("zz", "日本")])
def test_a_command_registered_on_japan_is_reachable(japan_group, name, arg):
    def handler(value: str = typer.Argument(...)):
        typer.echo(f"{name}:{value}")

    japan_group.command(name)(handler)
    app, _ = _app_and_japan()

    result = runner.invoke(app, ["japan", name, arg])
    assert result.exit_code == 0
    assert f"{name}:{arg}" in result.output

    listing = runner.invoke(app, ["japan", "--help"])
    assert name in listing.output
    # It lives under japan, not at the root.
    assert runner.invoke(app, [name, arg]).exit_code != 0


def test_registration_does_not_leak_between_tests():
    _, japan = _app_and_japan()
    assert all(c.name not in {"ping", "echo-back", "zz"} for c in japan.registered_commands)


def test_cli_runs_as_a_process_and_reports_the_version():
    env = {**os.environ, "PYTHONPATH": str(ROOT / "src")}
    code = "import sys; sys.argv[0] = 'horsetrack'; from horsetrack.cli import app; app()"
    proc = subprocess.run(
        [sys.executable, "-c", code, "--version"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr
    assert horsetrack.__version__ in proc.stdout
    proc = subprocess.run(
        [sys.executable, "-c", code, "japan", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=60,
    )
    assert proc.returncode == 0, proc.stderr


def test_pyproject_declares_the_console_script_and_typer():
    text = (ROOT / "pyproject.toml").read_text()
    scripts = re.search(r"^\[project\.scripts\]\n(.*?)(?=^\[|\Z)", text, re.S | re.M)
    assert scripts, "no [project.scripts] table"
    assert re.search(r"""^horsetrack\s*=\s*["']horsetrack\.cli:app["']""", scripts.group(1), re.M)
    deps = re.search(r"^dependencies\s*=\s*\[(.*?)\]", text, re.S | re.M)
    assert deps and re.search(r"""["']typer\b""", deps.group(1), re.I)
