"""A cloudflared quick tunnel to the board server, so a phone can load it (ticket #505).

`quick_tunnel(port)` starts `cloudflared tunnel --url http://127.0.0.1:<port> --no-autoupdate`
(cloudflared found on PATH), reads its output until the `https://<name>.trycloudflare.com` URL
appears and then on until the first registered connection (a `Registered tunnel connection` line:
the banner comes before the tunnel can serve), and only then, if cloudflared still runs, yields
that URL. The output is drained unread for as long as the child runs, so it never blocks on a full
pipe and nothing it logs later reaches the terminal. When the block exits, normally or by an
exception, KeyboardInterrupt included, the child is terminated and killed if it still runs
`STOP_GRACE_S` later. A quick-tunnel URL is public: the server behind it gates every path with an
access token (`server.make_server(token=...)`).
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
# api.trycloudflare.com is the service cloudflared asks for a tunnel, and login.trycloudflare.com
# the protected tunnel's auth broker: never the tunnel itself
_URL = re.compile(
    r"https://(?!api\.|login\.)[a-z0-9](?:[a-z0-9-]*[a-z0-9])?\.trycloudflare\.com(?![\w.-])"
)
# "Unregistered tunnel connection" and "Registered session" are not one
_REGISTERED = re.compile(r"\bRegistered tunnel connection\b")


class TunnelError(RuntimeError):
    """cloudflared is missing, exited, or printed no tunnel URL or registered connection in
    time."""


def parse_tunnel_url(output: str) -> str | None:
    """The quick-tunnel URL `https://<name>.trycloudflare.com` in cloudflared's output, else
    None."""
    match = _URL.search(output)
    return match.group(0) if match else None


class _Output(threading.Thread):
    """Reads the child's output to its end: line by line until the URL and then a registered
    connection, unread after them."""

    def __init__(self, stream: IO[bytes]):
        super().__init__(name="cloudflared-output", daemon=True)
        self.stream = stream
        self.url: str | None = None
        self.registered = False
        self.tail: collections.deque[str] = collections.deque(maxlen=3)
        self.ready = threading.Event()  # set once a connection registered or the output ended
        self.stopping = False

    def run(self) -> None:
        try:
            for raw in self.stream:
                if not self.registered:
                    self._read(raw.decode("utf-8", "replace").strip())
        except (OSError, ValueError):  # the pipe closed under us
            pass
        finally:
            self.ready.set()
            if self.registered and not self.stopping:
                log.warning("cloudflared exited: the tunnel is down and the phone URL is dead")

    def _read(self, line: str) -> None:
        if line:
            self.tail.append(line[:200])
        if self.url is None:
            self.url = parse_tunnel_url(line)
        elif _REGISTERED.search(line):
            self.registered = True
            self.ready.set()


def _stop(child: subprocess.Popen[bytes], output: _Output | None) -> None:
    """Terminate the child, kill it if it still runs after STOP_GRACE_S, and reap it; `output` is
    None or unstarted when its reader never got going."""
    if output is not None:
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
        if output is not None and output.is_alive():  # never join a reader that did not start
            output.join(timeout=1)
        if (output is None or not output.is_alive()) and child.stdout is not None:
            child.stdout.close()  # no reader owns it any more, or none ever did


@contextmanager
def quick_tunnel(port: int, *, timeout_s: float = 30.0) -> Iterator[str]:
    """Run a cloudflared quick tunnel to http://127.0.0.1:`port` and yield its public URL once a
    connection registered. TunnelError when cloudflared is not on PATH, exits before printing a URL
    or before a connection registered, has exited by the time the URL would be yielded, or did not
    do both within `timeout_s`; no child is left running in any case."""
    exe = shutil.which("cloudflared")
    if exe is None:
        raise TunnelError(
            "cloudflared is not on PATH: install it (brew install cloudflared) or serve without "
            "--tunnel"
        )
    argv = [exe, "tunnel", "--url", f"http://127.0.0.1:{port}", "--no-autoupdate"]
    child: subprocess.Popen[bytes] | None = None
    output: _Output | None = None
    try:  # a child, once spawned, is reaped even if its reader never starts
        try:
            child = subprocess.Popen(
                argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
            )
        except OSError as exc:
            raise TunnelError(f"cloudflared did not start: {exc}") from exc
        output = _Output(child.stdout)
        output.start()
        ended = output.ready.wait(timeout_s)  # one wait: the URL and a registration both fit
        reason = None
        if not output.registered or output.url is None:
            if output.url is None:
                what, missed = "printing a tunnel URL", "printed no tunnel URL"
            else:
                what, missed = "a tunnel connection registered", "registered no tunnel connection"
            if ended:  # the output ended
                reason = f"cloudflared exited before {what}"
            else:
                reason = f"cloudflared {missed} within {timeout_s:g} s"
        elif child.poll() is not None:  # registered, then gone before the handoff
            reason = "cloudflared exited after a tunnel connection registered, before the handoff"
        if reason is not None:
            last = " | ".join(output.tail)
            raise TunnelError(f"{reason}: {last}" if last else reason)
        yield output.url
    finally:
        if child is not None:
            _stop(child, output)
