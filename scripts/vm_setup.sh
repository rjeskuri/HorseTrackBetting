#!/usr/bin/env bash
# Set the Japan odds logger up on the Linux VM (ticket #33): Ubuntu with systemd; the trial VM is
# an aarch64 (Ampere) host, and nothing here is specific to one architecture.
#
#   sudo bash vm_setup.sh --repo-url URL --tag TAG [--user NAME] [--dry-run]
#
# As root, in this order: install uv into /usr/local/bin when no uv is on PATH (uv's own
# installer, which picks the build for this machine); clone URL into /opt/horsetrack, or fetch its
# tags when that checkout exists, and check TAG out (detached); `uv sync --frozen --no-dev` there,
# with any Python uv downloads kept under /opt/uv-python so a non-root service user can run the
# venv; create the data dir /var/lib/horsetrack (owned by --user); write the service and the timer
# from that checkout's `horsetrack japan systemd --print` into /etc/systemd/system; reload systemd
# and `enable --now` the timer. It never starts the service itself: the timer does, every 30
# minutes. Run it again with another tag to upgrade the checkout and the units.
#
# --user NAME   run the logger as NAME, an existing user (the unit's User=); without it, as root.
# --dry-run     print every command on stdout, in order, and run none; no root needed.
#
# It never writes or reads a .env: the owner writes /opt/horsetrack/.env on the VM (the user
# agent; readable by the service user), and the unit reads it.

set -euo pipefail
umask 022

INSTALL_DIR=/opt/horsetrack
PYTHON_DIR=/opt/uv-python
DATA_DIR=/var/lib/horsetrack
UNIT_DIR=/etc/systemd/system
SERVICE=horsetrack-japan-logger.service
TIMER=horsetrack-japan-logger.timer
UV_INSTALLER=https://astral.sh/uv/install.sh
UV_BIN_DIR=/usr/local/bin

die() {
  printf 'vm_setup.sh: %s\n' "$*" >&2
  exit 2
}

usage() {
  printf '%s\n' "usage: vm_setup.sh --repo-url URL --tag TAG [--user NAME] [--dry-run]" >&2
}

# ------------------------------------------------------------------ arguments, before anything

repo_url=
tag=
user=
dry_run=0
while [ $# -gt 0 ]; do
  case $1 in
    --repo-url | --tag | --user)
      if [ $# -lt 2 ]; then die "$1 needs a value"; fi
      case $2 in -*) die "$1 needs a value, not $2" ;; esac
      case $1 in
        --repo-url) repo_url=$2 ;;
        --tag) tag=$2 ;;
        --user) user=$2 ;;
      esac
      shift 2
      ;;
    --dry-run)
      dry_run=1
      shift
      ;;
    -h | --help)
      usage
      exit 0
      ;;
    *)
      usage
      die "unknown argument: $1"
      ;;
  esac
done

[ -n "$tag" ] || die "--tag TAG is required: the git tag to check out"
case $tag in *[!A-Za-z0-9._/+-]*) die "--tag $tag: not a plain tag name" ;; esac
[ -n "$repo_url" ] || die "--repo-url URL is required"
case $repo_url in *[[:space:]\'\"]*) die "--repo-url: the URL has a space or a quote in it" ;; esac
case $user in *[!A-Za-z0-9_-]*) die "--user $user: not a user name" ;; esac

# ------------------------------------------------------------------ printing and running

q() { # one word as the shell reads it, quoted only when it has to be (no word holds a quote)
  case $1 in
    '' | *[!A-Za-z0-9_./:=+,-]*) printf "'%s'" "$1" ;;
    *) printf '%s' "$1" ;;
  esac
}

planned() { # print the command line $1; true when it is to run (no --dry-run)
  if [ "$dry_run" = 1 ]; then
    printf '%s\n' "$1"
    return 1
  fi
  printf '+ %s\n' "$1" >&2
}

run() { # print the command "$@"; run it unless --dry-run
  local line="" word
  for word in "$@"; do line="$line${line:+ }$(q "$word")"; done
  if planned "$line"; then "$@"; fi
}

unit_text() { # $1: a file name; its text in the --print output $units (after its header line)
  printf '%s\n' "$units" | awk -v want="# $1" -v a="# $SERVICE" -v b="# $TIMER" '
    { head = $0; sub(/[ \t\r]+$/, "", head) }
    head == a || head == b { on = (head == want); next }
    on { print }'
}

write_unit() { # $1: a file name, $2: its text
  if planned "tee $UNIT_DIR/$1 >/dev/null  # its text after the line \"# $1\""; then
    printf '%s\n' "$2" | tee "$UNIT_DIR/$1" >/dev/null
  fi
}

# ------------------------------------------------------------------ the host (real run only)

if [ "$dry_run" = 1 ]; then
  printf '%s\n' "# dry run: the commands a real run makes as root, in order; none has run"
else
  [ "$(id -u)" = 0 ] || die "run it as root: sudo bash $0 ..."
  [ "$(uname -s)" = Linux ] || die "the VM runs Linux with systemd, not $(uname -s)"
  for tool in git awk tee systemctl; do
    command -v "$tool" >/dev/null 2>&1 || die "$tool is not installed"
  done
  if [ -n "$user" ] && ! id -u "$user" >/dev/null 2>&1; then
    die "--user $user: no such user (create it first)"
  fi
fi

# ------------------------------------------------------------------ the steps

if ! command -v uv >/dev/null 2>&1; then
  into="env UV_INSTALL_DIR=$UV_BIN_DIR UV_NO_MODIFY_PATH=1 sh"
  if planned "curl --proto =https --tlsv1.2 -LsSf $UV_INSTALLER | $into"; then
    command -v curl >/dev/null 2>&1 || die "curl is not installed (needed for uv's installer)"
    curl --proto =https --tlsv1.2 -LsSf "$UV_INSTALLER" |
      env UV_INSTALL_DIR="$UV_BIN_DIR" UV_NO_MODIFY_PATH=1 sh
    command -v uv >/dev/null 2>&1 || die "uv went into $UV_BIN_DIR, which is not on PATH"
  fi
fi

if [ -d "$INSTALL_DIR/.git" ]; then
  run git -C "$INSTALL_DIR" fetch --tags --force "$repo_url"
elif [ -e "$INSTALL_DIR" ]; then
  die "$INSTALL_DIR exists and is not a git checkout"
else
  run git clone "$repo_url" "$INSTALL_DIR"
fi
run git -C "$INSTALL_DIR" -c advice.detachedHead=false checkout --detach "refs/tags/$tag"
run env UV_PYTHON_INSTALL_DIR="$PYTHON_DIR" uv --directory "$INSTALL_DIR" sync --frozen --no-dev

run mkdir -p "$DATA_DIR"
if [ -n "$user" ]; then run chown "$user:" "$DATA_DIR"; fi

set -- japan systemd --print --repo-dir "$INSTALL_DIR" --data-dir "$DATA_DIR"
if [ -n "$user" ]; then set -- "$@" --user "$user"; fi
units=
line="$INSTALL_DIR/.venv/bin/horsetrack"
for word in "$@"; do line="$line $(q "$word")"; done
if planned "$line"; then
  units=$("$INSTALL_DIR/.venv/bin/horsetrack" "$@")
  service_text=$(unit_text "$SERVICE")
  timer_text=$(unit_text "$TIMER")
  case $service_text in *'[Service]'*) ;; *) die "no $SERVICE in the --print output" ;; esac
  case $timer_text in *'[Timer]'*) ;; *) die "no $TIMER in the --print output" ;; esac
fi
write_unit "$SERVICE" "${service_text:-}"
write_unit "$TIMER" "${timer_text:-}"

run systemctl daemon-reload
run systemctl enable --now "$TIMER"

if [ "$dry_run" != 1 ]; then
  printf '%s\n' "set up $TIMER at $tag: it starts $SERVICE at :00 and :30, Tokyo time" >&2
fi
