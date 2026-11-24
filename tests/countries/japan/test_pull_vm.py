"""Pulling the VM's logs back to the Mac (ticket #34).

Contract pinned here (module `horsetrack.countries.japan.logger.pull_vm`):
- `pull_vm(host, remote_dir="/var/lib/horsetrack", dry_run=False, runner=None) -> int` runs one
  rsync per subtree through `runner(argv: list[str]) -> int` (the exit code) and returns 0 when
  every call returned 0, else a non-zero code. `runner=None` means the module attribute
  `run_rsync` (looked up at call time, so tests may monkeypatch it; the default really runs rsync).
- One call per subtree, each `rsync -a --partial [--dry-run] <host>:<remote_dir>/<subtree>/ <dest>`
  with dest `<data_dir>/vm/<host>/<subtree>/`, for the subtrees `odds_snapshots`,
  `raw/jra/official_odds` and `logs`. Transport is plain ssh by the alias: no `-e` other than
  `ssh`, no key, password, user or address anywhere in argv; never a `--delete*` or
  `--remove-source-files` option. The destination directories exist when the runner is called
  (rsync only creates the last path component), unless it is a dry run.
- `host` must be an ssh alias: an IPv4/IPv6 address, `user@host`, anything with `@`, `:`, `/`,
  whitespace, a leading `-`, or empty raises ValueError and the runner is never called.
- `launchd_plist(host) -> str`: plist XML text for a LaunchAgent that runs
  `... horsetrack japan pull-vm --host <host>` (the program path before `horsetrack` is not
  pinned) daily at 22:00 local: StartCalendarInterval Hour 22, Minute 0, no Weekday/Day/Month.
  The Label differs per host.
- CLI: `horsetrack japan pull-vm --host ALIAS [--remote-dir D] [--dry-run] [--print-launchd]`.
  `--print-launchd` prints the plist to stdout and does not run rsync. A refused host or a failed
  rsync exits non-zero.

All fixtures are synthetic and built in tmp_path.
"""

from __future__ import annotations

import plistlib
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from horsetrack.core import paths

DEFAULT_REMOTE = "/var/lib/horsetrack"
SUBTREES = ["odds_snapshots", "raw/jra/official_odds", "logs"]
ALIASES = ["vm1", "horse-trial", "trial_vm_2"]
IPV4 = re.compile(r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")


@pytest.fixture(autouse=True)
def data_dir(tmp_path, monkeypatch) -> Path:
    """The data directory under tmp_path; no real `.env` can supply a setting."""
    root = tmp_path / "repo"
    root.mkdir()
    monkeypatch.setattr("horsetrack.core.env.repo_root", lambda: root)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(tmp_path / "data"))
    (tmp_path / "data").mkdir()
    return tmp_path / "data"


class Recorder:
    """A fake rsync runner: records argv and what existed on disk at call time."""

    def __init__(self, code: int = 0):
        self.code = code
        self.calls: list[list[str]] = []
        self.dest_existed: list[bool] = []

    def __call__(self, argv):
        self.calls.append(list(argv))
        self.dest_existed.append(Path(argv[-1]).is_dir())
        return self.code


def pull(*args, **kwargs):
    """The unit under test, imported per call so each test fails on its own while it is missing."""
    from horsetrack.countries.japan.logger.pull_vm import pull_vm as impl

    return impl(*args, **kwargs)


def plist_text(host):
    from horsetrack.countries.japan.logger.pull_vm import launchd_plist as impl

    return impl(host)


def run_cli(*args):
    from horsetrack.cli import app

    return CliRunner().invoke(app, ["japan", "pull-vm", *args])


def dest_of(argv) -> Path:
    return Path(argv[-1])


def source_of(argv) -> str:
    return argv[-2]


# ------------------------------------------------------------------ the pull


@pytest.mark.parametrize("host", ALIASES)
@pytest.mark.parametrize("remote", [DEFAULT_REMOTE, "/srv/ht", "/srv/ht/"])
def test_pull_targets_the_vm_subtree_and_never_deletes(data_dir, host, remote):
    kwargs = {} if remote == DEFAULT_REMOTE else {"remote_dir": remote}
    runner = Recorder()
    assert pull(host, runner=runner, **kwargs) == 0

    base = remote.rstrip("/")
    assert len(runner.calls) == len(SUBTREES)
    by_source = {source_of(argv): argv for argv in runner.calls}
    assert set(by_source) == {f"{host}:{base}/{sub}/" for sub in SUBTREES}

    vm_root = data_dir / "vm" / host
    for sub in SUBTREES:
        argv = by_source[f"{host}:{base}/{sub}/"]
        assert argv[0] == "rsync"
        assert "-a" in argv and "--partial" in argv
        # one destination per subtree, under the VM's own subtree, never merged or on the Mac's logs
        assert dest_of(argv) == vm_root / sub
        assert dest_of(argv).is_relative_to(data_dir / "vm" / host)
        # nothing deletes: no --delete*, no --remove-source-files
        assert not [a for a in argv if a.startswith("--delete") or a == "--remove-source-files"]
        assert "--dry-run" not in argv and "-n" not in argv
    assert all(runner.dest_existed)

    # the Mac's own logs are never a destination and are not created
    for own in ("logs", "odds_snapshots", "raw"):
        assert not (data_dir / own).exists()
    assert {p.name for p in data_dir.iterdir()} == {"vm"}


def test_different_hosts_get_different_subtrees(data_dir):
    a, b = Recorder(), Recorder()
    pull("vm1", runner=a)
    pull("vm2", runner=b)
    assert {dest_of(c).parts[len(data_dir.parts) + 1] for c in a.calls} == {"vm1"}
    assert {dest_of(c).parts[len(data_dir.parts) + 1] for c in b.calls} == {"vm2"}
    assert all(source_of(c).startswith("vm2:") for c in b.calls)


def test_the_default_runner_is_the_module_attribute(monkeypatch, data_dir):
    from horsetrack.countries.japan.logger import pull_vm as mod

    rec = Recorder()
    monkeypatch.setattr(mod, "run_rsync", rec)
    assert pull("vm1") == 0
    assert len(rec.calls) == len(SUBTREES)


def test_no_key_password_or_address_in_the_arguments(data_dir):
    runner = Recorder()
    pull("vm1", remote_dir="/srv/ht", dry_run=True, runner=runner)
    for argv in runner.calls:
        remote_args = [a for a in argv if a not in ("rsync",) and not a.startswith(str(data_dir))]
        joined = " ".join(remote_args)
        assert "@" not in joined
        assert not IPV4.search(joined)
        assert not re.search(
            r"\bsshpass\b|password|passwd|identity|\.pem|id_rsa|id_ed25519", joined, re.I
        )
        assert "-i" not in argv
        assert not [a for a in argv if a.startswith("-i")]
        # transport is plain ssh by alias: any -e / --rsh value is exactly `ssh`
        for i, a in enumerate(argv):
            if a == "-e" or a == "--rsh":
                assert argv[i + 1] == "ssh"
            if a.startswith("--rsh="):
                assert a == "--rsh=ssh"
            assert not a.startswith("-p") and not a.startswith("--port")


def test_dry_run_is_passed_to_rsync(data_dir):
    runner = Recorder()
    assert pull("vm1", dry_run=True, runner=runner) == 0
    assert len(runner.calls) == len(SUBTREES)
    for argv in runner.calls:
        assert argv.count("--dry-run") == 1
        assert argv[0] == "rsync" and "-a" in argv and "--partial" in argv
        assert not [a for a in argv if a.startswith("--delete")]
        assert dest_of(argv).is_relative_to(data_dir / "vm" / "vm1")

    live = Recorder()
    pull("vm1", dry_run=False, runner=live)
    assert all("--dry-run" not in argv for argv in live.calls)


def test_a_failing_rsync_gives_a_non_zero_result(data_dir):
    assert pull("vm1", runner=Recorder(code=23)) != 0
    assert pull("vm1", runner=Recorder(code=0)) == 0


# ------------------------------------------------------------------ the host


@pytest.mark.parametrize(
    "host",
    [
        "1.2.3.4",
        "user@1.2.3.4",
        "root@vm1",
        "192.168.0.10",
        "::1",
        "2001:db8::1",
        "[2001:db8::1]",
        "vm1:/etc",
        "vm1/../x",
        "vm 1",
        "-oProxyCommand=evil",
        "",
    ],
)
def test_the_host_must_be_an_alias_not_an_address(host, data_dir):
    runner = Recorder()
    with pytest.raises(ValueError):
        pull(host, runner=runner)
    assert runner.calls == []
    assert not (data_dir / "vm").exists()


@pytest.mark.parametrize("host", ["1.2.3.4", "user@1.2.3.4"])
def test_the_cli_refuses_an_address(host, data_dir, monkeypatch):
    from horsetrack.countries.japan.logger import pull_vm as mod

    rec = Recorder()
    monkeypatch.setattr(mod, "run_rsync", rec)
    result = run_cli("--host", host)
    assert result.exit_code != 0
    assert rec.calls == []
    assert not (data_dir / "vm").exists()


# ------------------------------------------------------------------ the CLI


def test_cli_pulls_and_passes_options_through(data_dir, monkeypatch):
    from horsetrack.countries.japan.logger import pull_vm as mod

    rec = Recorder()
    monkeypatch.setattr(mod, "run_rsync", rec)

    result = run_cli("--host", "vm1")
    assert result.exit_code == 0, result.output
    assert len(rec.calls) == len(SUBTREES)
    assert {source_of(c) for c in rec.calls} == {f"vm1:{DEFAULT_REMOTE}/{s}/" for s in SUBTREES}
    assert all("--dry-run" not in c for c in rec.calls)

    rec.calls.clear()
    result = run_cli("--host", "vm2", "--remote-dir", "/srv/ht", "--dry-run")
    assert result.exit_code == 0, result.output
    assert {source_of(c) for c in rec.calls} == {f"vm2:/srv/ht/{s}/" for s in SUBTREES}
    assert all(c.count("--dry-run") == 1 for c in rec.calls)
    assert all(dest_of(c).is_relative_to(data_dir / "vm" / "vm2") for c in rec.calls)


def test_cli_exits_non_zero_when_rsync_fails(data_dir, monkeypatch):
    from horsetrack.countries.japan.logger import pull_vm as mod

    monkeypatch.setattr(mod, "run_rsync", Recorder(code=12))
    assert run_cli("--host", "vm1").exit_code != 0


def test_cli_requires_a_host(data_dir):
    result = run_cli()
    assert result.exit_code == 2
    assert "--host" in result.output


# ------------------------------------------------------------------ launchd


@pytest.mark.parametrize("host", ALIASES)
def test_print_launchd_runs_daily_at_22_00(host, data_dir, monkeypatch):
    from horsetrack.countries.japan.logger import pull_vm as mod

    rec = Recorder()
    monkeypatch.setattr(mod, "run_rsync", rec)

    for text in (plist_text(host), run_cli("--host", host, "--print-launchd").stdout):
        plist = plistlib.loads(text.encode())
        assert isinstance(plist["Label"], str) and plist["Label"]

        interval = plist["StartCalendarInterval"]
        assert isinstance(interval, dict)
        assert interval["Hour"] == 22 and interval["Minute"] == 0
        # daily: nothing narrows it to a weekday, day of month or month
        assert set(interval) <= {"Hour", "Minute"}

        args = plist["ProgramArguments"]
        assert args[-4:] == ["japan", "pull-vm", "--host", host]
        assert Path(args[-5]).name == "horsetrack" or args[-5] == "horsetrack"
        assert "--dry-run" not in args
        assert "@" not in " ".join(args) and not IPV4.search(" ".join(args))
        assert "StartInterval" not in plist
    assert rec.calls == []  # printing the plist runs no rsync


def test_launchd_labels_differ_per_host():
    a = plistlib.loads(plist_text("vm1").encode())
    b = plistlib.loads(plist_text("vm2").encode())
    assert a["Label"] != b["Label"]


def test_launchd_refuses_an_address():
    with pytest.raises(ValueError):
        plist_text("user@1.2.3.4")
    with pytest.raises(ValueError):
        plist_text("1.2.3.4")


# ------------------------------------------------------------------ launchd: env, logs, remote dir


def _plists(host, **kwargs):
    """The plist from the function and from the CLI, parsed; both must agree on what is pinned."""
    from horsetrack.countries.japan.logger.pull_vm import launchd_plist as impl

    cli_args = ["--host", host, "--print-launchd"]
    if "remote_dir" in kwargs:
        cli_args += ["--remote-dir", kwargs["remote_dir"]]
    cli = run_cli(*cli_args)
    assert cli.exit_code == 0, cli.output
    return [plistlib.loads(impl(host, **kwargs).encode()), plistlib.loads(cli.stdout.encode())]


@pytest.mark.parametrize("host", ALIASES)
def test_launchd_pins_the_data_dir_in_the_job_environment(host, data_dir):
    for plist in _plists(host):
        env = plist["EnvironmentVariables"]
        assert env[paths.DATA_DIR_VAR] == str(data_dir.resolve())


def test_launchd_environment_follows_the_data_dir(tmp_path, monkeypatch):
    other = tmp_path / "elsewhere" / "ht-data"
    other.mkdir(parents=True)
    monkeypatch.setenv(paths.DATA_DIR_VAR, str(other))
    for plist in _plists("vm1"):
        assert plist["EnvironmentVariables"][paths.DATA_DIR_VAR] == str(other.resolve())


@pytest.mark.parametrize("host", ALIASES)
def test_launchd_job_output_goes_to_the_vm_subtree_not_the_mac_logs(host, data_dir):
    root = data_dir.resolve()
    for plist in _plists(host):
        for key in ("StandardOutPath", "StandardErrorPath"):
            target = Path(plist[key])
            assert target.is_absolute()
            assert target.is_relative_to(root / "vm" / host), (key, target)
            assert not target.is_relative_to(root / "logs"), (key, target)


def test_launchd_log_paths_differ_per_host(data_dir):
    a, _ = _plists("vm1")
    b, _ = _plists("vm2")
    assert Path(a["StandardOutPath"]).parent != Path(b["StandardOutPath"]).parent
    assert Path(b["StandardErrorPath"]).is_relative_to(data_dir.resolve() / "vm" / "vm2")


def test_launchd_without_a_data_dir_raises_and_the_cli_prints_nothing(monkeypatch):
    from horsetrack.countries.japan.logger.pull_vm import launchd_plist as impl

    monkeypatch.delenv(paths.DATA_DIR_VAR, raising=False)  # fixture already hides any `.env`
    with pytest.raises(paths.DataDirNotSet):
        impl("vm1")
    result = run_cli("--host", "vm1", "--print-launchd")
    assert result.exit_code != 0
    assert result.stdout.strip() == ""
    assert "<plist" not in result.output


@pytest.mark.parametrize("host", ["vm1", "horse-trial"])
@pytest.mark.parametrize("remote", ["/srv/ht", "/mnt/data/horsetrack"])
def test_launchd_carries_a_non_default_remote_dir(host, remote):
    for plist in _plists(host, remote_dir=remote):
        args = plist["ProgramArguments"]
        i = args.index("--remote-dir")
        assert args[i + 1] == remote
        assert args.count("--remote-dir") == 1
        assert args[args.index("--host") + 1] == host
        assert "--dry-run" not in args


def test_launchd_remote_dir_differs_per_call():
    (a, _), (b, _) = _plists("vm1", remote_dir="/srv/a"), _plists("vm1", remote_dir="/srv/b")
    assert a["ProgramArguments"] != b["ProgramArguments"]


# ------------------------------------------------------------------ launchd: the log folder exists
# launchd does not create the folder of StandardOutPath / StandardErrorPath: a job whose log folder
# is missing fails to start (exit 78) and never runs. So `--print-launchd` creates that folder.


def _tree(root: Path) -> set[Path]:
    return {p.relative_to(root) for p in root.rglob("*")}


@pytest.mark.parametrize("host", ALIASES)
def test_print_launchd_creates_the_folder_of_the_job_logs(host, data_dir, monkeypatch):
    from horsetrack.countries.japan.logger import pull_vm as mod

    rec = Recorder()
    monkeypatch.setattr(mod, "run_rsync", rec)
    assert not (data_dir / "vm").exists()

    result = run_cli("--host", host, "--print-launchd")
    assert result.exit_code == 0, result.output
    plist = plistlib.loads(result.stdout.encode())
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert Path(plist[key]).parent.is_dir(), key
    assert Path(plist["StandardOutPath"]).parent == data_dir.resolve() / "vm" / host
    assert rec.calls == []  # still no rsync


def test_print_launchd_creates_only_the_one_hosts_folder(data_dir):
    result = run_cli("--host", "vm1", "--print-launchd")
    assert result.exit_code == 0, result.output
    # nothing but `vm/vm1` (no logs/, no other host, no rsync subtree, no log file itself)
    assert _tree(data_dir) == {Path("vm"), Path("vm/vm1")}
    assert not (data_dir / "logs").exists()

    result = run_cli("--host", "horse-trial", "--print-launchd")
    assert result.exit_code == 0, result.output
    assert _tree(data_dir) == {Path("vm"), Path("vm/vm1"), Path("vm/horse-trial")}


def test_print_launchd_keeps_what_is_already_in_the_log_folder(data_dir):
    folder = data_dir / "vm" / "vm1"
    (folder / "logs").mkdir(parents=True)
    (folder / "pull-vm.err.log").write_text("last night\n")
    (folder / "logs" / "a.log").write_text("kept\n")

    for _ in range(2):  # a second print is fine too
        assert run_cli("--host", "vm1", "--print-launchd").exit_code == 0
    assert (folder / "pull-vm.err.log").read_text() == "last night\n"
    assert (folder / "logs" / "a.log").read_text() == "kept\n"


@pytest.mark.parametrize("host", ["1.2.3.4", "user@1.2.3.4", "vm1:/etc", "-oProxyCommand=evil"])
def test_print_launchd_with_a_refused_host_creates_nothing(host, data_dir):
    result = run_cli("--host", host, "--print-launchd")
    assert result.exit_code != 0
    assert "<plist" not in result.stdout
    assert _tree(data_dir) == set()


def test_print_launchd_without_a_data_dir_creates_nothing(tmp_path, monkeypatch):
    workdir = tmp_path / "cwd"
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.delenv(paths.DATA_DIR_VAR, raising=False)
    result = run_cli("--host", "vm1", "--print-launchd")
    assert result.exit_code != 0
    assert _tree(workdir) == set()  # no relative `vm/` either
    assert _tree(tmp_path / "data") == set()
    assert _tree(tmp_path / "repo") == set()
