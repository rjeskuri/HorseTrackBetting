"""The public-tree guard: `scripts/check_public_tree.py` keeps data, secrets and personal data out.

Contract pinned here (the coder follows it):

* `check(paths, read) -> list[Violation]`; each Violation has `path` (exactly as given to check),
  `rule` and `detail` (a str). The rule ids are the names below, one per rule (a)-(f) of #16.
* Size: 256 KB is 256 * 1024 bytes, measured as `len(read(path))`; exactly that size passes.
* A matched secret shows in a detail and in the printed output only as its first 4 characters
  then `***`.
* `main()` lists the files with `git ls-files` in the current directory (the repo root), reads
  them relative to it, prints one line per violation on stdout (each naming the path and the
  rule, nothing else on stdout) and exits 1, or prints nothing and exits 0.

Every input that looks like a secret, a home path, an email address or an artifact link is
assembled at run time from parts, so this file passes its own tracked-tree test.
"""

from __future__ import annotations

import importlib
import os
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
SCRIPT = SCRIPTS / "check_public_tree.py"

DATA, FIXTURE, SIZE, JRDB, SECRET, PERSONAL = (
    "data",
    "fixture",
    "size",
    "jrdb",
    "secret",
    "personal",
)
LIMIT = 256 * 1024
MARKER = "<!-- synthetic:"

# Parts, so no literal in this file is itself a violation.
AT = "@"
MAC_HOME = "/Us" + "ers/"
LINUX_HOME = "/ho" + "me/"
CLAUDE_AI = "claude" + ".ai/"
DASHES = "-" * 5
GH_NOREPLY = "users.noreply" + ".github.com"
ANTHROPIC_NOREPLY = "noreply" + AT + "anthropic.com"
KEY_BODY = "MIIEpAIBAAKCAQEA" + "Q7wZ" * 10
UUID = "3fa85f64-5717-4562-b3fc-2c963f66afa6"

# Files this ticket adds: checked by the tracked-tree test even before they are committed.
OWN_FILES = (
    "tests/test_public_tree.py",
    "scripts/check_public_tree.py",
    "tests/fixtures/synthetic/README.md",
)


@pytest.fixture(scope="module")
def cpt():
    sys.path.insert(0, str(SCRIPTS))
    try:
        return importlib.import_module("check_public_tree")
    finally:
        sys.path.remove(str(SCRIPTS))


def run(cpt, files: dict[str, bytes]):
    return cpt.check(list(files), files.__getitem__)


def by_path(violations) -> dict[str, set[str]]:
    out: dict[str, set[str]] = defaultdict(set)
    for v in violations:
        out[v.path].add(v.rule)
    return out


def assert_refused_exactly(cpt, files: dict[str, bytes], refused: set[str], rule: str) -> None:
    """Every refused path is reported under `rule`; every other path is not reported at all."""
    found = by_path(run(cpt, files))
    missed = sorted(p for p in refused if rule not in found.get(p, set()))
    assert not missed, f"not reported under {rule!r}: {missed}"
    wrongly = {p: sorted(r) for p, r in found.items() if p not in refused}
    assert not wrongly, f"clean input reported: {wrongly}"


def key_header(kind: str) -> str:
    return DASHES + "BEGIN " + kind + "PRIV" + "ATE KEY" + DASHES


def tokens() -> dict[str, str]:
    """One token per named family, each with a distinct body so a hard-coded answer cannot pass."""
    return {
        "ghp": "gh" + "p_" + "Q7wZ" * 9,
        "gho": "gh" + "o_" + "Rt5Y" * 9,
        "github_pat": "github" + "_pat_" + "11AB7CDEF0" + "Hj3k" * 3 + "_" + ("Kx9m" * 15)[:59],
        "aws": "AK" + "IA" + "Q7WZ" * 4,
        "xoxb": "xo" + "xb-" + "1234567890-1234567890123-" + "Lp8n" * 6,
        "xoxp": "xo" + "xp-" + "2234567890-2234567890-2234567890123-" + "a1b2c3d4" * 4,
        "xoxa": "xo" + "xa-" + "2-3234567890-3234567890-" + "Mn4b" * 8,
        "xoxs": "xo" + "xs-" + "4234567890-4234567890-" + "Vc6x" * 8,
        "xoxr": "xo" + "xr-" + "5234567890-5234567890-" + "Zq2w" * 8,
    }


def reveals_more_than_four(secret: str, text: str) -> bool:
    """The text shows the secret past its first 4 characters: the next 8, or its last 4."""
    return secret[4:12] in text or secret[-4:] in text


def git(cwd: Path, *args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args], cwd=cwd, env=clean_env(), capture_output=True, text=True, check=True
    )


def clean_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def run_script(cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(SCRIPT)],
        cwd=cwd,
        env=clean_env(),
        capture_output=True,
        text=True,
        encoding="utf-8",
        timeout=60,
    )


def test_the_tracked_tree_is_clean(cpt):
    tracked = [p for p in git(REPO, "ls-files", "-z").stdout.split("\0") if p]
    paths = [p for p in tracked if (REPO / p).is_file()]
    paths += [p for p in OWN_FILES if p not in paths and (REPO / p).is_file()]
    assert "tests/test_public_tree.py" in paths and "uv.lock" in paths

    violations = cpt.check(paths, lambda p: (REPO / p).read_bytes())
    assert violations == [], [(v.path, v.rule, v.detail) for v in violations]

    proc = run_script(REPO)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert proc.stdout.strip() == ""


def test_a_tracked_data_path_is_refused(cpt):
    assert cpt.check([], lambda p: b"") == []
    files = {
        "data/races.csv": b"race,horse\n1,2\n",
        "data/raw/jra/results/2025/202508040611.html.gz": b"\x1f\x8b\x08\x00",
        "data/.gitkeep": b"",
        "data/odds_snapshots/jra/2026-11-22/KYO.jsonl.gz": b"\x1f\x8b",
        "src/horsetrack/core/data_x.py": b"X = 1\n",
        "docs/data.md": b"# Data\nData lives under $HORSETRACK_DATA_DIR, never data/ in git.\n",
        "database/schema.sql": b"create table races (id int);\n",
        "tests/core/test_data.py": b"def test_x():\n    pass\n",
        "metadata.json": b"{}\n",
        ".gitignore": b"/data/\n.env\n",
    }
    refused = {p for p in files if p.startswith("data/")}
    assert_refused_exactly(cpt, files, refused, DATA)


def test_a_fixture_outside_synthetic_is_refused(cpt):
    marked = (MARKER + " hand-built -->\n<html></html>\n").encode()
    files = {
        "tests/fixtures/race.json": b'{"race": 1}\n',
        "tests/fixtures/jra/results.html.gz": b"\x1f\x8b\x08\x00",
        "tests/fixtures/page.html": marked,
        "tests/fixtures/synthetic_old/race.json": b"{}\n",
        "tests/fixtures/real/synthetic/race.json": b"{}\n",
        "tests/fixtures/hkjc/2025/card.csv": b"a,b\n",
        "tests/fixtures/synthetic/README.md": b"# Synthetic fixtures\n",
        "tests/fixtures/synthetic/jra/race.json": b'{"race": 2}\n',
        "tests/fixtures/synthetic/jra/deep/card.csv": b"a,b\n",
        "tests/fixtures/synthetic/jra/ok.html": marked,
        "tests/fixtures.py": b"X = 1\n",
        "tests/conftest.py": b"import pytest\n",
        "tests/core/test_fixtures_loader.py": b"def test_x():\n    pass\n",
    }
    refused = {
        "tests/fixtures/race.json",
        "tests/fixtures/jra/results.html.gz",
        "tests/fixtures/page.html",
        "tests/fixtures/synthetic_old/race.json",
        "tests/fixtures/real/synthetic/race.json",
        "tests/fixtures/hkjc/2025/card.csv",
    }
    assert_refused_exactly(cpt, files, refused, FIXTURE)


def test_an_html_fixture_without_the_synthetic_marker_is_refused(cpt):
    base = "tests/fixtures/synthetic/"
    ok = (MARKER + " hand-built race card -->\n<html><body>race</body></html>\n").encode()
    files = {
        base + "jra/race.html": b"<html><body>race</body></html>\n",
        base + "jra/card.htm": b"<!DOCTYPE html>\n<html></html>\n",
        base + "late.html": b"<html>\n" + ok,
        base + "nocolon.html": b"<!-- synthetic -->\n<html></html>\n",
        base + "other.htm": b"<!-- real: copied page -->\n<html></html>\n",
        base + "empty.html": b"",
        base + "nested/deep/page.htm": b"<html></html>\n",
        base + "jra/ok.html": ok,
        base + "jra/ok.htm": ok,
        base + "nested/deep/ok.html": ok,
        base + "jra/race.json": b'{"race": 1}\n',
        base + "jra/notes.txt": b"<html> is mentioned here\n",
        base + "README.md": b"# Synthetic fixtures\n",
    }
    refused = {
        base + "jra/race.html",
        base + "jra/card.htm",
        base + "late.html",
        base + "nocolon.html",
        base + "other.htm",
        base + "empty.html",
        base + "nested/deep/page.htm",
    }
    assert_refused_exactly(cpt, files, refused, FIXTURE)


def test_a_large_file_is_refused_but_uv_lock_is_allowed(cpt):
    lock = b'version = 1\nrequires-python = ">=3.10"\n'
    lock += b'[[package]]\nname = "pydantic"\nversion = "2.14.4"\n' * (LIMIT // 40 * 3)
    assert len(lock) > LIMIT
    files = {
        "assets/one_over.bin": b"\0" * (LIMIT + 1),
        "docs/huge.md": b"some words of text\n" * (LIMIT // 19 * 4),
        "uv.lock.bak": lock,
        "backup-uv.lock": lock,
        "assets/at_limit.bin": b"\0" * LIMIT,
        "assets/just_under.bin": b"\0" * (LIMIT - 1),
        "uv.lock": lock,
        "small.txt": b"small\n",
    }
    refused = {"assets/one_over.bin", "docs/huge.md", "uv.lock.bak", "backup-uv.lock"}
    assert_refused_exactly(cpt, files, refused, SIZE)

    # The exemption is from the size rule only: uv.lock's content is still checked.
    secret = tokens()["aws"]
    leaky = {"uv.lock": lock + f'token = "{secret}"\n'.encode()}
    found = by_path(run(cpt, leaky))
    assert found.get("uv.lock") == {SECRET}


def test_a_jrdb_style_file_name_is_refused(cpt):
    files = {
        "KYI150712.txt": b"x\n",
        "notes/jrdb/SED250105.txt": b"x\n",
        "src/horsetrack/countries/japan/jrdb/samples/UKC991231.txt": b"x\n",
        "tests/fixtures/synthetic/jrdb/BAC201231.txt": b"x\n",
        "KYI15071.txt": b"x\n",
        "KYI1507123.txt": b"x\n",
        "KYI150712.csv": b"x\n",
        "KY150712.txt": b"x\n",
        "docs/jrdb.md": b"KYI150712.txt is the horse-index file; SED250105.txt the results.\n",
        "src/horsetrack/countries/japan/jrdb/kyi.py": b"NAME = 'KYI'\n",
    }
    refused = {
        "KYI150712.txt",
        "notes/jrdb/SED250105.txt",
        "src/horsetrack/countries/japan/jrdb/samples/UKC991231.txt",
        "tests/fixtures/synthetic/jrdb/BAC201231.txt",
    }
    assert_refused_exactly(cpt, files, refused, JRDB)


def test_secret_patterns_are_refused_and_masked(cpt):
    toks = tokens()
    kinds = {"rsa": "RSA ", "openssh": "OPENSSH ", "ec": "EC ", "pkcs8": "", "enc": "ENCRYPTED "}
    headers = {name: key_header(kind) for name, kind in kinds.items()}
    headers["pgp"] = DASHES + "BEGIN PGP " + "PRIV" + "ATE KEY BLOCK" + DASHES
    files: dict[str, bytes] = {}
    for family, tok in toks.items():
        files[f"cfg/{family}.txt"] = f'TOKEN = "{tok}"\n'.encode()
    for name, header in headers.items():
        end = header.replace("BEGIN", "END")
        files[f"keys/{name}.pem"] = f"{header}\n{KEY_BODY}\n{end}\n".encode()
    files["blobs/cache.bin"] = b"\xff\xfe\x00" + toks["aws"].encode() + b"\x00\x80"
    refused = set(files)

    files["docs/secrets.md"] = (
        b"GitHub tokens start with ghp_, gho_ or github_pat_; Slack tokens with xoxb- or "
        b"xoxp-; AWS keys with AKIA. Never commit a private key.\n"
    )
    files["keys/public.pem"] = f"{DASHES}BEGIN PUBLIC KEY{DASHES}\n{KEY_BODY}\n".encode()
    files["keys/cert.pem"] = f"{DASHES}BEGIN CERTIFICATE{DASHES}\n{KEY_BODY}\n".encode()
    files["keys/pgp_public.asc"] = f"{DASHES}BEGIN PGP PUBLIC KEY BLOCK{DASHES}\n".encode()
    files["cfg/near_aws.txt"] = ("AK" + "IA" + ("Q7WZ" * 4)[:15] + " is one short\n").encode()
    files["uv.lock"] = ('hash = "sha256:' + "ab12" * 16 + '"\n').encode()
    files["blobs/clean.bin"] = b"\x89PNG\r\n\x1a\n" + bytes(range(128, 256))
    assert_refused_exactly(cpt, files, refused, SECRET)

    violations = run(cpt, files)
    every_secret = [*toks.values(), *headers.values(), KEY_BODY]
    for v in violations:
        assert isinstance(v.detail, str)
        for s in every_secret:
            assert s not in v.detail and s not in v.path, (v.path, v.detail)
    details = defaultdict(list)
    for v in violations:
        if v.rule == SECRET:
            details[v.path].append(v.detail)
    for family, tok in toks.items():
        for d in details[f"cfg/{family}.txt"]:
            assert tok[:4] + "***" in d, (family, d)
            assert not reveals_more_than_four(tok, d), (family, d)
    for d in details["blobs/cache.bin"]:
        assert toks["aws"][:4] + "***" in d, d
        assert not reveals_more_than_four(toks["aws"], d), d
    for name in headers:
        assert all("***" in d for d in details[f"keys/{name}.pem"]), details[f"keys/{name}.pem"]


def test_home_paths_and_personal_emails_are_refused(cpt):
    refused_text = {
        "docs/mac.md": f"cd {MAC_HOME}alice/Projects/HorseTrackBetting\n",
        "scripts/linux.sh": f'export X="{LINUX_HOME}bob/data/"\n',
        "docs/file_url.md": f"see file://{MAC_HOME}j.doe/notes/x.md\n",
        "docs/runner.txt": f"{LINUX_HOME}runner/work/repo/\n",
        "docs/plain_email.md": "write to alice.smith" + AT + "example.com today\n",
        "docs/angle_email.md": "Bob <bob+racing" + AT + "mail.example.org>\n",
        "docs/mailto.md": "[mail](mailto:Carol.Jones" + AT + "Example.CO.UK)\n",
        "docs/anthropic_person.md": "someone" + AT + "anthropic.com\n",
        "docs/other_noreply.md": "noreply" + AT + "example.com\n",
        "docs/prefixed_noreply.md": "x" + ANTHROPIC_NOREPLY + "\n",
        "docs/suffix_trick.md": "x" + AT + GH_NOREPLY + ".example.org\n",
        "docs/lookalike.md": "x" + AT + "not" + GH_NOREPLY + "\n",
    }
    clean_text = {
        "tests/core/test_paths_like.py": 'r"' + MAC_HOME + '",\nr"' + LINUX_HOME + '",\n',
        ".env.example": "HORSETRACK_DATA_DIR=~/Data/horsetrack\n",
        "docs/tools.md": "/usr/local/bin/python3, $HOME/Data and ~/Projects/HKTrack\n",
        ".github/workflows/ci.yml": "uses: actions/checkout@v7\nuses: astral-sh/setup-uv@v10.3.0\n",
        "package.json": '{"devDependencies": {"@types/node": "^20", "react@18.3.1": "x"}}\n',
        "pyproject.toml": 'dependencies = ["pydantic>=2.14.4"]\n',
        "tests/core/test_deco.py": "@pytest.fixture\ndef x():\n    return a @ b\n",
        ".githooks/pre-commit": 'ruff "$@"\n',
    }
    files = {p: t.encode() for p, t in {**refused_text, **clean_text}.items()}
    assert_refused_exactly(cpt, files, set(refused_text), PERSONAL)


def test_claude_ai_artifact_urls_are_refused(cpt):
    refused_text = {
        "docs/share.md": "https://" + CLAUDE_AI + "artifact/" + UUID.replace("-", "") + "\n",
        "docs/code.md": "[page](https://" + CLAUDE_AI + "code/artifact/" + UUID + ")\n",
        "docs/public.md": "https://" + CLAUDE_AI + "public/artifacts/" + UUID + "\n",
        "docs/bare.md": "see " + CLAUDE_AI + "artifact/0f1e2d3c4b5a69788796a5b4c3d2e1f0\n",
    }
    clean_text = {
        "docs/attribution.md": "Generated with [Claude Code](https://claude.com/claude-code)\n",
        "docs/notes.md": "The bump is an artifact of the split, not a signal.\n",
    }
    files = {p: t.encode() for p, t in {**refused_text, **clean_text}.items()}
    assert_refused_exactly(cpt, files, set(refused_text), PERSONAL)


def test_noreply_addresses_are_allowed(cpt):
    clean_text = {
        "docs/gh_id.md": "12345678+someone" + AT + GH_NOREPLY + "\n",
        "docs/gh_plain.md": "someone" + AT + GH_NOREPLY + "\n",
        "scripts/bot.py": 'BOT = "41898282+github-actions[bot]' + AT + GH_NOREPLY + '"\n',
        "docs/coauthor.md": "Co-Authored-By: Claude Opus 5.5 <" + ANTHROPIC_NOREPLY + ">\n",
        "docs/rule.md": "emails other than " + AT + GH_NOREPLY + " and " + ANTHROPIC_NOREPLY + "\n",
        "docs/both.md": ANTHROPIC_NOREPLY + ", 1+a" + AT + GH_NOREPLY + ", b" + AT + GH_NOREPLY,
    }
    files = {p: t.encode() for p, t in clean_text.items()}
    assert run(cpt, files) == []

    # An allowed address does not excuse a personal one beside it.
    files["docs/mixed.md"] = (ANTHROPIC_NOREPLY + " and alice" + AT + "example.com\n").encode()
    assert_refused_exactly(cpt, files, {"docs/mixed.md"}, PERSONAL)


def test_the_command_prints_one_line_per_violation_and_exits_1(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    toks = tokens()
    lock = b"version = 1\n" + b"# pad\n" * (LIMIT // 6 + 10)
    tracked = {
        "README.md": b"# demo\n",
        "uv.lock": lock,
        "docs/team.md": ("Co-Authored-By: Claude <" + ANTHROPIC_NOREPLY + ">\n").encode(),
        "data/races.csv": b"race,horse\n",
        "tests/fixtures/race.json": b"{}\n",
        "tests/fixtures/synthetic/page.html": b"<html></html>\n",
        "notes/KYI150712.txt": b"x\n",
        "assets/big.bin": b"\0" * (LIMIT + 1),
        "config/aws.txt": f"key = {toks['aws']}\n".encode(),
        "deploy/token.sh": f"export T={toks['ghp']}\n".encode(),
        "docs/setup.md": f"cd {MAC_HOME}alice/Projects/x\n".encode(),
        "docs/contact.md": ("mail alice" + AT + "example.com\n").encode(),
        "docs/share.md": ("https://" + CLAUDE_AI + "artifact/" + "ab" * 16 + "\n").encode(),
        "docs/競馬メモ.md": f"{LINUX_HOME}bob/notes/\n".encode(),
    }
    expected = {
        "data/races.csv": DATA,
        "tests/fixtures/race.json": FIXTURE,
        "tests/fixtures/synthetic/page.html": FIXTURE,
        "notes/KYI150712.txt": JRDB,
        "assets/big.bin": SIZE,
        "config/aws.txt": SECRET,
        "deploy/token.sh": SECRET,
        "docs/setup.md": PERSONAL,
        "docs/contact.md": PERSONAL,
        "docs/share.md": PERSONAL,
        "docs/競馬メモ.md": PERSONAL,
    }
    untracked = {
        "scratch/notes.md": f"cd {MAC_HOME}carol/x/\n".encode(),
        "data/untracked.csv": b"x\n",
    }
    for rel, body in {**tracked, **untracked}.items():
        (repo / rel).parent.mkdir(parents=True, exist_ok=True)
        (repo / rel).write_bytes(body)
    git(repo, "init", "-q")
    git(repo, "add", "--", *tracked)

    proc = run_script(repo)
    assert proc.returncode == 1, proc.stdout + proc.stderr
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert len(lines) == len(expected), lines
    for line in lines:
        named = [p for p in expected if p in line]
        assert len(named) == 1, (line, named)
        assert expected[named[0]] in line, line
    assert sorted(p for line in lines for p in expected if p in line) == sorted(expected)
    for rel in untracked:
        assert rel not in proc.stdout
    output = proc.stdout + proc.stderr
    for tok in (toks["aws"], toks["ghp"]):
        assert tok not in output
        assert not reveals_more_than_four(tok, output)
    aws_line = next(ln for ln in lines if "config/aws.txt" in ln)
    assert toks["aws"][:4] + "***" in aws_line
    ghp_line = next(ln for ln in lines if "deploy/token.sh" in ln)
    assert toks["ghp"][:4] + "***" in ghp_line
