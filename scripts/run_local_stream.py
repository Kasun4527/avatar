# SPDX-FileCopyrightText: 2026 Goodsize Inc.
# SPDX-License-Identifier: LicenseRef-AVTR-1-Community

"""Top-level orchestrator: renderer subprocess + health-poll + streamer.

Spawns the avtr1_renderer FastAPI app under the renderer env (`pixi run -e default
python -m avtr1_renderer.api.app`), polls `GET /health` until it returns 200,
then spawns the localrtc streamer (`python -m avaturn_live_streamer.local_stream_cli`).
Both children inherit env so AVTR1_LOCAL_STORAGE / CLOUDFLARE_TURN_* propagate
from the parent shell. Conversation-engine credentials (OpenAI / Cartesia API
keys) are entered per-session in the local-stream UI and stored only in the
browser's localStorage, not env. The streamer's renderer wiring is injected here
(mode=single, lb_or_instance_url=http://localhost:{RENDERER_PORT}) so no backend
env file is read.

Run via the streamer env:
    pixi run -e streamer python scripts/run_local_stream.py
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from contextlib import suppress
from pathlib import Path
from typing import Callable

import httpx
from dotenv import load_dotenv

LOG = logging.getLogger("orchestrator")

# Load .env into os.environ before anything else reads env vars.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    return int(raw) if raw else default


def _start_renderer(port: int) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env.setdefault("LOAD_BALANCER_URL", "disabled")
    cmd = [
        "pixi", "run", "-e", "default",
        "python", "-m", "uvicorn",
        "avtr1_renderer.api.app:app",
        "--host", "0.0.0.0",
        "--port", str(port),
    ]
    LOG.info("starting renderer: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


class _Interrupted(Exception):
    pass


def _wait_for_health(
    port: int,
    timeout_s: float = 300.0,
    is_interrupted: Callable[[], bool] = lambda: False,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_err: str | None = None
    with httpx.Client(timeout=3.0) as client:
        while time.monotonic() < deadline:
            if is_interrupted():
                raise _Interrupted()
            try:
                r = client.get(f"http://localhost:{port}/health")
                if r.status_code == 200:
                    LOG.info("renderer healthy")
                    return
                last_err = f"HTTP {r.status_code}: {r.text[:200]}"
            except httpx.HTTPError as e:
                last_err = f"{type(e).__name__}: {e}"
            slept = 0.0
            while slept < 1.0 and not is_interrupted():
                time.sleep(0.1)
                slept += 0.1
    raise TimeoutError(f"renderer /health did not become 200 within {timeout_s:.0f}s; last error: {last_err}")


def _start_streamer(host: str, port: int, renderer_port: int) -> subprocess.Popen[bytes]:
    env = os.environ.copy()
    env["RENDERERS__RENDERERS__AVTRN_1__MODE"] = "single"
    env["RENDERERS__RENDERERS__AVTRN_1__LB_OR_INSTANCE_URL"] = f"http://localhost:{renderer_port}"
    idle_timeout = _env_int("IDLE_TIMEOUT", 300)
    max_duration = _env_int("MAX_DURATION", 7200)
    cmd = [
        sys.executable, "-m", "avaturn_live_streamer.local_stream_cli",
        "--host", host,
        "--port", str(port),
        "--idle-timeout", str(idle_timeout),
        "--max-duration", str(max_duration),
    ]
    LOG.info("starting streamer: %s", " ".join(cmd))
    return subprocess.Popen(cmd, env=env)


def _terminate(proc: subprocess.Popen[bytes], name: str, grace_s: float = 20.0) -> None:
    if proc.poll() is not None:
        return
    LOG.info("terminating %s (pid=%d)", name, proc.pid)
    with suppress(ProcessLookupError):
        proc.terminate()
    try:
        proc.wait(timeout=grace_s)
    except subprocess.TimeoutExpired:
        LOG.warning("%s did not exit gracefully; killing", name)
        with suppress(ProcessLookupError):
            proc.kill()
        proc.wait()


def main() -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s orchestrator: %(message)s",
    )

    host = os.environ.get("STREAMER_HOST", "0.0.0.0")
    port = _env_int("STREAMER_PORT", 7860)
    renderer_port = _env_int("RENDERER_PORT", 8000)

    renderer = _start_renderer(renderer_port)
    streamer: subprocess.Popen[bytes] | None = None

    interrupted = False

    def _sig_handler(signum: int, _frame: object) -> None:
        nonlocal interrupted
        LOG.info("orchestrator received signal %d; shutting down", signum)
        interrupted = True

    signal.signal(signal.SIGINT, _sig_handler)
    signal.signal(signal.SIGTERM, _sig_handler)

    try:
        try:
            _wait_for_health(renderer_port, is_interrupted=lambda: interrupted)
        except _Interrupted:
            LOG.info("interrupted during renderer warmup; stopping renderer")
            return 130
        except TimeoutError as e:
            LOG.error(str(e))
            return 1
        if renderer.poll() is not None:
            LOG.error("renderer exited with code %d before becoming healthy", renderer.returncode)
            return renderer.returncode or 1

        streamer = _start_streamer(host, port, renderer_port)
        LOG.info("streamer started pid=%d", streamer.pid)

        while not interrupted:
            time.sleep(0.5)
            if renderer.poll() is not None:
                LOG.error("renderer exited (code=%d) — stopping streamer", renderer.returncode)
                break
            if streamer.poll() is not None:
                LOG.error("streamer exited (code=%d) — stopping renderer", streamer.returncode)
                break

        rc = streamer.returncode if streamer and streamer.returncode is not None else 0
        if renderer.returncode is not None and renderer.returncode != 0:
            rc = renderer.returncode
        return rc
    finally:
        if streamer is not None:
            _terminate(streamer, "streamer")
        _terminate(renderer, "renderer")


if __name__ == "__main__":
    sys.exit(main())