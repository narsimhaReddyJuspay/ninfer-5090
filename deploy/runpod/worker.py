"""RunPod serverless worker for NInfer.

Starts one resident `ninfer-serve` process per worker and proxies each job to its
OpenAI-compatible HTTP endpoint. The engine (paged KV, prefix reuse, CUDA graphs,
1-8 request decode batch) stays warm for the lifetime of the worker, so only the
first job after a cold start pays model load; RunPod keeps the worker alive until
the endpoint's idle timeout expires.

The engine begins loading at worker boot (NINFER_PRELOAD=0 defers it to the
first job), and GPU/CPU utilization is logged to stdout every
NINFER_METRICS_INTERVAL_S seconds (0 disables), which surfaces in the RunPod
worker logs.

Job input is an OpenAI chat-completions body (see docs/serving.md):

    {"messages": [...], "max_tokens": 256, "enable_thinking": false}

Optionally wrapped as {"route": "/v1/chat/completions", "payload": {...}} to hit a
different route. "stream": true streams SSE chunks back as job output parts.
"""

import json
import os
import shlex
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request

import runpod

PORT = int(os.environ.get("NINFER_PORT", "8080"))
MODEL = os.environ.get("NINFER_MODEL", "/models/qwen3_8_27b.ninfer")
SERVE_BIN = os.environ.get("NINFER_SERVE", "/usr/local/bin/ninfer-serve")
LOAD_TIMEOUT_S = int(os.environ.get("NINFER_LOAD_TIMEOUT_S", "900"))
REQUEST_TIMEOUT_S = int(os.environ.get("NINFER_REQUEST_TIMEOUT_S", "600"))
METRICS_INTERVAL_S = float(os.environ.get("NINFER_METRICS_INTERVAL_S", "30"))
GPU_INDEX = int(os.environ.get("NINFER_GPU_INDEX", "0"))

# Confirmed ninfer-serve flags (docs/serving.md); anything else goes through
# NINFER_SERVE_ARGS, e.g. `--prefill-chunk 1024 --kv-dtype int8`.
_FLAG_ENV = {
    "NINFER_MAX_CONTEXT": "--max-context",
    "NINFER_KV_CAPACITY": "--kv-capacity",
    "NINFER_MAX_CONCURRENCY": "--max-concurrency",
    "NINFER_SPEC": "--spec",
    "NINFER_DRAFT_TOKENS": "--draft-tokens",
    "NINFER_MODEL_ID": "--model-id",
    "NINFER_API_KEY": "--api-key",  # unset/empty => engine serves without auth
}
_DEFAULTS = {
    "NINFER_MAX_CONTEXT": "8192",
    "NINFER_MAX_CONCURRENCY": "8",
    "NINFER_SPEC": "mtp",
    "NINFER_DRAFT_TOKENS": "3",
}

# serve_metrics.py ships next to this file in the worker image; from a repo
# checkout it lives at tools/bench/serve_metrics.py.
_HERE = os.path.dirname(os.path.abspath(__file__))
for _candidate in (_HERE, os.path.join(_HERE, "..", "..", "tools", "bench")):
    if os.path.isfile(os.path.join(_candidate, "serve_metrics.py")):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
        break

_serve_proc = None
_start_lock = threading.Lock()
_ready = threading.Event()
_preload_attempted = False
_start_error = None


def _serve_cmd():
    cmd = [SERVE_BIN, MODEL, "--host", "127.0.0.1", "--port", str(PORT)]
    for env_name, flag in _FLAG_ENV.items():
        value = os.environ.get(env_name, _DEFAULTS.get(env_name))
        if value is not None:
            cmd += [flag, value]
    if os.environ.get("NINFER_LM_HEAD_DRAFT", "1") == "1":
        cmd.append("--lm-head-draft")
    cmd += shlex.split(os.environ.get("NINFER_SERVE_ARGS", ""))
    return cmd


def _port_open():
    try:
        with socket.create_connection(("127.0.0.1", PORT), timeout=2):
            return True
    except OSError:
        return False


def _post(route, payload, timeout):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:2000]
        return {"error": {"status": error.code, "message": detail}}


def _warmup():
    probe = {
        "messages": [{"role": "user", "content": "ping"}],
        "max_tokens": 1,
        "enable_thinking": False,
    }
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        result = _post("/v1/chat/completions", probe, timeout=60)
        if "error" not in result:
            return
        if not _port_open():
            break
        time.sleep(2)
    raise RuntimeError("ninfer-serve started but never answered the warmup probe")


def ensure_running():
    """Start ninfer-serve once per worker; restart only if it died."""
    global _serve_proc, _start_error
    with _start_lock:
        if _serve_proc is not None and _serve_proc.poll() is None and _port_open():
            _start_error = None
            return
        started = time.monotonic()
        try:
            # Inherit this process's stdout/stderr: RunPod captures container
            # logs, and a PIPE that is never drained would deadlock the engine
            # once its log output fills the OS pipe buffer (~64 KB).
            _serve_proc = subprocess.Popen(_serve_cmd())
        except OSError as error:
            raise RuntimeError(
                f"failed to launch ninfer-serve at {SERVE_BIN}: {error}"
            ) from error
        while time.monotonic() - started < LOAD_TIMEOUT_S:
            if _port_open():
                _warmup()
                _start_error = None
                return
            if _serve_proc.poll() is not None:
                raise RuntimeError(
                    f"ninfer-serve exited with code {_serve_proc.returncode} "
                    f"(engine output is in the worker logs above)"
                )
            time.sleep(1)
        raise RuntimeError(
            f"ninfer-serve did not become ready within {LOAD_TIMEOUT_S}s "
            f"(model={MODEL}; check the volume mount and artifact path)"
        )


def _metrics_loop(interval):
    try:
        from serve_metrics import CpuTracker, format_sample, read_once
    except ImportError:
        return
    tracker = CpuTracker()
    read_once(GPU_INDEX, tracker)  # prime the CPU delta
    while True:
        time.sleep(interval)
        try:
            print(f"[metrics] {format_sample(read_once(GPU_INDEX, tracker))}", flush=True)
        except Exception as error:  # keep the loop alive; a bad sample is not fatal
            print(f"[metrics] error: {error!r}", flush=True)


def _bootstrap():
    """Boot path: log the host GPU/driver, load the engine, keep metrics alive."""
    global _start_error
    try:
        from serve_metrics import read_driver

        driver = read_driver(GPU_INDEX)
        if driver:
            # CUDA 13.1 needs an r580+ driver; this line makes a mismatch visible.
            print(
                f"[worker] host: {driver['name']} driver={driver['driver_version']}",
                flush=True,
            )
    except ImportError:
        pass
    try:
        ensure_running()
        print(f"[worker] engine ready: {MODEL}", flush=True)
    except Exception as error:
        _start_error = error
        print(f"[worker] engine failed to start: {error}", flush=True)
    else:
        if METRICS_INTERVAL_S > 0:
            threading.Thread(
                target=_metrics_loop, args=(METRICS_INTERVAL_S,), daemon=True
            ).start()
    finally:
        _ready.set()


def handler(job):
    if _preload_attempted and not _ready.is_set():
        if not _ready.wait(timeout=LOAD_TIMEOUT_S + 120):
            raise RuntimeError("engine still loading from a cold start; retry the job")
    body = job.get("input") or {}
    route = "/v1/chat/completions"
    if "payload" in body:
        route = body.get("route", route)
        body = body["payload"]
    # A boot-time failure may be transient (slow volume mount); every job gets a
    # fresh ensure_running() attempt instead of failing forever on a stale error.
    ensure_running()
    if body.get("stream"):
        return _stream(route, body)
    return _post(route, body, timeout=REQUEST_TIMEOUT_S)


def _stream(route, payload):
    request = urllib.request.Request(
        f"http://127.0.0.1:{PORT}{route}",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        response = urllib.request.urlopen(request, timeout=REQUEST_TIMEOUT_S)
    except urllib.error.HTTPError as error:
        # Relay engine 4xx/5xx as a first chunk so the client sees the status and
        # message instead of a job-level traceback.
        detail = error.read().decode("utf-8", "replace")[:2000]
        yield {"error": {"status": error.code, "message": detail}}
        return
    with response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                yield json.loads(data)
            except ValueError:
                continue


if os.environ.get("NINFER_PRELOAD", "1") == "1":
    _preload_attempted = True
    threading.Thread(target=_bootstrap, daemon=True).start()

runpod.serverless.start({"handler": handler})
