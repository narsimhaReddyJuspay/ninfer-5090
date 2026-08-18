#!/usr/bin/env python3
"""Local OpenAI-compatible bridge to a RunPod serverless endpoint running the
NInfer worker (deploy/runpod/worker.py).

RunPod serverless workers have no inbound HTTP, so clairvoyance's OpenAI client
cannot reach the engine directly. This bridge exposes the ordinary OpenAI HTTP
surface locally and forwards each request as a RunPod job: non-streaming
requests relay the job's final output, and `stream: true` relays the worker's
yielded chunks as server-sent events. The engine stays resident on the worker;
the bridge adds no GPU cost.

  pip install runpod
  export RUNPOD_API_KEY=...
  python3 deploy/runpod/bridge.py --endpoint-id <id> --port 8081

  # then from clairvoyance / anywhere:
  #   OpenAI(base_url="http://127.0.0.1:8081/v1", api_key="-")

Requires RUNPOD_API_KEY in the environment; the endpoint id may also be given
via RUNPOD_ENDPOINT_ID.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, Optional, Sequence

import runpod

ENDPOINT_ID: Optional[str] = None
MODEL_NAME = "ninfer"
_endpoint: Optional[Any] = None
_endpoint_lock = threading.Lock()


def endpoint():
    global _endpoint
    with _endpoint_lock:
        if _endpoint is None:
            _endpoint = runpod.Endpoint(ENDPOINT_ID)
        return _endpoint


class BridgeHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ninfer-runpod-bridge/1"

    def log_message(self, fmt: str, *args: Any) -> None:
        sys.stderr.write(
            f"[bridge] {self.address_string()} {self.command} {self.path} "
            f"{fmt % args}\n"
        )

    def _send_json(self, status: int, body: Dict[str, Any]) -> None:
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path in ("/health", "/healthz"):
            self._send_json(200, {"status": "ok", "endpoint_id": ENDPOINT_ID})
            return
        if self.path == "/v1/models":
            self._send_json(
                200,
                {
                    "object": "list",
                    "data": [{"id": MODEL_NAME, "object": "model", "owned_by": "ninfer"}],
                },
            )
            return
        self._send_json(404, {"error": {"message": f"unknown route {self.path}"}})

    def do_POST(self) -> None:
        if not self.path.startswith("/v1/"):
            self._send_json(404, {"error": {"message": f"unknown route {self.path}"}})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            body = json.loads(self.rfile.read(length) or b"{}")
            if not isinstance(body, dict):
                raise ValueError("request body must be a JSON object")
        except (ValueError, json.JSONDecodeError) as error:
            self._send_json(400, {"error": {"message": f"invalid JSON body: {error}"}})
            return

        # The worker accepts {"route", "payload"}; default route mirrors this path.
        job_input: Dict[str, Any] = {"route": self.path, "payload": body}
        try:
            job = endpoint().run(job_input)
        except Exception as error:  # SDK/network failures before a job exists
            self._send_json(502, {"error": {"message": f"runpod submit failed: {error}"}})
            return

        if body.get("stream"):
            self._relay_stream(job)
        else:
            self._relay_final(job)

    def _relay_stream(self, job: Any) -> None:
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        try:
            for chunk in job.stream():
                if not isinstance(chunk, dict):
                    continue
                self.wfile.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            return
        except Exception as error:
            tail = json.dumps({"error": {"message": str(error)}}).encode()
            self.wfile.write(b"data: " + tail + b"\n\n")
            self.wfile.flush()
        try:
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _relay_final(self, job: Any) -> None:
        try:
            output = job.output()
        except Exception as error:
            self._send_json(502, {"error": {"message": f"runpod job failed: {error}"}})
            return
        if isinstance(output, dict) and "error" in output:
            status = output["error"].get("status", 502) if isinstance(output["error"], dict) else 502
            self._send_json(int(status) if status >= 400 else 502, output)
            return
        if output is None:
            self._send_json(502, {"error": {"message": "worker returned no output"}})
            return
        self._send_json(200, output)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--endpoint-id", default=os.environ.get("RUNPOD_ENDPOINT_ID"),
                        help="RunPod serverless endpoint id (or RUNPOD_ENDPOINT_ID)")
    parser.add_argument("--port", type=int, default=8081)
    parser.add_argument("--model-name", default=os.environ.get("NINFER_MODEL_ID", "ninfer"),
                        help="model id advertised at /v1/models")
    args = parser.parse_args(argv)

    global ENDPOINT_ID, MODEL_NAME
    if not args.endpoint_id:
        print("error: --endpoint-id or RUNPOD_ENDPOINT_ID is required", file=sys.stderr)
        return 2
    ENDPOINT_ID = args.endpoint_id
    MODEL_NAME = args.model_name

    server = ThreadingHTTPServer(("0.0.0.0", args.port), BridgeHandler)
    print(f"bridge listening on http://0.0.0.0:{args.port}/v1 -> runpod endpoint {ENDPOINT_ID}")
    print(f"openai base_url: http://127.0.0.1:{args.port}/v1 (model '{MODEL_NAME}')")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
