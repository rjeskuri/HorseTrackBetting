"""A cloudflared quick tunnel to the board server, so a phone can load it (ticket #505).

`quick_tunnel(port)` starts `cloudflared tunnel --url http://127.0.0.1:<port> --no-autoupdate`
(cloudflared found on PATH), reads its output until the `https://<name>.trycloudflare.com` URL
appears and yields that URL. The output is drained unread for as long as the child runs, so it
never blocks on a full pipe and nothing it logs later reaches the terminal. When the block exits,
normally or by an exception, KeyboardInterrupt included, the child is terminated and killed if it
still runs `STOP_GRACE_S` later. A quick-tunnel URL is public: the server behind it gates every
path with an access token (`server.make_server(token=...)`).
"""

from __future__ import annotations

import collections
import logging
import re
import shutil
import subprocess
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from typing import IO

log = logging.getLogger(__name__)

STOP_GRACE_S = 5.0
# api.trycloudflare.com is the service cloudflared asks for a tunnel, never the tunnel itself
_URL = re.compile(r"https://(?!api\.)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com(?![\w.-])")


class TunnelError(RuntimeError):
    """cloudflared is missing, exited, or printed no tunnel URL in time."""


def parse_tunnel_url(output: str) -> str | None:
    """The quick-tunnel URL `https://<name>.trycloudflare.com` in cloudflared's output, else
    None."""
    match = _URL.search(output)
    return match.group(0) if match else None


class _Output(threading.Thread):
    """Reads the child's output to its end: line by line until the URL, unread after it."""

    def __init__(self, stream: IO[bytes]):
        super().__init__(name="cloudflared-output", daemon=True)
        self.stream = stream
        self.url: str | None = None
        self.tail: collections.deque[str] = collections.deque(maxlen=3)
        self.ready = threading.Event()  # set once the URL was found or the output ended
        self.stopping = False

    def run(self) -> None:
        try:
            for raw in self.stream:
                if self.url is None:
                    line = raw.decode("utf-8", "replace").strip()
                    if line:
                        self.tail.append(line[:200])
                    self.url = parse_tunnel_url(line)
                    if self.url is not None:
                        self.ready.set()
        except (OSError, ValueError):  # the pipe closed under us
            pass
        finally:
            self.ready.set()
            if self.url is not None and not self.stopping:
                log.warning("cloudflared exited: the tunnel is down and the phone URL is dead")


def _stop(child: subprocess.Popen[bytes], output: _Output) -> None:
    """Terminate the child, kill it if it still runs after STOP_GRACE_S, and reap it."""
    output.stopping = True
    try:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(timeout=STOP_GRACE_S)
            except subprocess.TimeoutExpired:
                pass
    finally:  # also when a second interrupt cut the wait short
        if child.poll() is None:
            child.kill()
            child.wait()
        output.join(timeout=1)
        if not output.is_alive() and child.stdout is not None:
            child.stdout.close()


@contextmanager
def quick_tunnel(port: int, *, timeout_s: float = 30.0) -> Iterator[str]:
    """Run a cloudflared quick tunnel to http://127.0.0.1:`port` and yield its public URL.
    TunnelError when cloudflared is not on PATH, exits before printing a URL, or prints none
    within `timeout_s`; no child is left running in any case."""
    exe = shutil.which("cloudflared")
    if exe is None:
        raise TunnelError(
            "cloudflared is not on PATH: install it (brew install cloudflared) or serve without "
            "--tunnel"
        )
    argv = [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    try:
        child = subprocess.Popen(
            argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
    except OSError as exc:
        raise TunnelError(f"cloudflared did not start: {exc}") from exc
    output = _Output(child.stdout)
    output.start()
    try:
        ready = output.ready.wait(timeout_s)
        if output.url is None:
            if ready:  # the output ended
                reason = "cloudflared exited before printing a tunnel URL"
            else:
                reason = f"cloudflared printed no tunnel URL within {timeout_s:g} s"
            last = " | ".join(output.tail)
            raise TunnelError(f"{reason}: {last}" if last else reason)
        yield output.url
    finally:
        _stop(child, output)
