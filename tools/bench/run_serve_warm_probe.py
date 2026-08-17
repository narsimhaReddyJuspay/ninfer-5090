#!/usr/bin/env python3
"""Warm-path serving probe: TTFT after prefix warm-up, decode throughput, and
thinking-task latency against a running ninfer-serve (or its RunPod bridge).

Where ``tools/bench/run_serve_concurrency.py`` measures stock decode saturation,
this probe measures the clairvoyance-shaped path: a shared payments-domain
system prefix is prewarmed by the first request (prefix reuse is on by default),
and every later measurement is a warm request. When run on the GPU host, GPU and
CPU utilization are sampled for the whole campaign and summarized at the end.

Run against a locally serving engine:

  python3 -m tools.bench.run_serve_warm_probe \
      --base-url http://127.0.0.1:8080 --model qwen3.8-27b \
      --out probe_27b.json --metrics-csv probe_27b.csv

or through the RunPod bridge (deploy/runpod/bridge.py) with the same arguments.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench import serve_metrics  # noqa: E402

SYSTEM_PREFIX = (
    "You are the clairvoyance payments assistant for a large Indian payment processor. "
    "You answer questions about UPI, card settlements, chargebacks, refunds, and fraud "
    "using only the operational context below.\n\n"
    "Operational context: settlement files arrive twice daily at 06:00 and 18:00 IST and "
    "reconcile against the ledger within thirty minutes. UPI success is defined as credits "
    "posted to the beneficiary bank; pending transactions may take up to thirty minutes to "
    "reach a terminal state. Card chargebacks follow the issuer's representment window, "
    "typically forty-five days domestic and one hundred twenty days international. Refunds "
    "above rupees five lakh require a second approver before release. Fraud review queues "
    "are drained around the clock; a transaction frozen for review releases automatically "
    "after seventy-two hours unless an analyst escalates it. Peak traffic windows are "
    "lunchtime and evening rush; planned partner-bank maintenance is announced six hours "
    "ahead. Incident severity one requires a customer-facing status note within ten minutes.\n\n"
    "Answer concisely for an operations engineer. Prefer exact numbers and timelines from "
    "the context; say when something is not covered."
)
FILLER_SENTENCE = (
    "The daily recon report for merchant onboarding showed a small drift in captured "
    "settlement amounts, so the finance lead asked for the affected window and the "
    "reconciliation job to be rerun after the ledger sync finished. "
)
THINKING_TASKS = (
    "A merchant disputes a refund of rupees six lakh that was released without a second "
    "approver. Walk through the review steps, who must sign off retroactively, and what "
    "the customer-facing timeline is.",
    "UPI volume doubles during the evening rush while one partner bank is in a announced "
    "maintenance window. Reason about pending-transaction pressure, what to monitor, and "
    "when to publish a severity note.",
)


def post_json(base_url: str, route: str, payload: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    request = urllib.request.Request(
        base_url.rstrip("/") + route,
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as error:
        detail = error.read().decode("utf-8", "replace")[:1000]
        raise RuntimeError(f"HTTP {error.code} from {route}: {detail}") from error


def stream_chat(
    base_url: str, payload: Dict[str, Any], timeout: float
) -> Iterator[Tuple[Dict[str, Any], float]]:
    request = urllib.request.Request(
        base_url.rstrip("/") + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        for raw in response:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[len("data:"):].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except ValueError:
                continue
            yield chunk, time.monotonic()


def filler_to_target_tokens(target_tokens: int, already_tokens: int) -> str:
    # 1 token ~= 4 characters; repeat the sentence until it covers the remainder.
    remaining = max(0, target_tokens - already_tokens)
    repeats = int(remaining * 4 / len(FILLER_SENTENCE)) + 1
    return " ".join([FILLER_SENTENCE] * repeats)


def estimated_tokens(text: str) -> int:
    return len(text) // 4


def measure_ttft_ms(
    base_url: str, model: str, messages: List[Dict[str, str]], max_tokens: int
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0,
        "stream": True,
        "enable_thinking": False,
    }
    started = time.monotonic()
    first_token_s: Optional[float] = None
    content_chunks = 0
    total_chars = 0
    for chunk, arrived in stream_chat(base_url, payload, timeout=600):
        delta = (chunk.get("choices") or [{}])[0].get("delta", {}) if chunk.get("choices") else {}
        if delta.get("content"):
            content_chunks += 1
            total_chars += len(delta["content"])
            if first_token_s is None:
                first_token_s = arrived
    finished = time.monotonic()
    return {
        "ttft_ms": (first_token_s - started) * 1000.0 if first_token_s is not None else None,
        "stream_ms": (finished - started) * 1000.0,
        "content_chunks": content_chunks,
        "content_chars": total_chars,
    }


def decode_throughput(
    base_url: str, model: str, concurrency: int, decode_tokens: int, prefix_messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    payload = {
        "model": model,
        "messages": prefix_messages,
        "max_tokens": decode_tokens,
        "temperature": 0,
        "enable_thinking": False,
    }

    def one_request(_: int) -> Dict[str, Any]:
        started = time.monotonic()
        body = post_json(base_url, "/v1/chat/completions", payload, timeout=1800)
        elapsed = time.monotonic() - started
        usage = body.get("usage") or {}
        return {
            "elapsed_s": elapsed,
            "completion_tokens": usage.get("completion_tokens"),
            "prompt_tokens": usage.get("prompt_tokens"),
        }

    started = time.monotonic()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(one_request, range(concurrency)))
    wall_s = time.monotonic() - started
    token_counts = [r["completion_tokens"] for r in results if r["completion_tokens"]]
    aggregate = sum(token_counts) / wall_s if token_counts else None
    per_stream = [r["completion_tokens"] / r["elapsed_s"] for r in results if r["completion_tokens"]]
    return {
        "concurrency": concurrency,
        "wall_s": wall_s,
        "aggregate_tok_s": aggregate,
        "per_stream_tok_s_mean": (sum(per_stream) / len(per_stream)) if per_stream else None,
        "requests": results,
    }


def thinking_task(
    base_url: str, model: str, task: str, cap_tokens: int, prefix_messages: List[Dict[str, str]]
) -> Dict[str, Any]:
    messages = prefix_messages + [{"role": "user", "content": task}]
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": cap_tokens,
        "temperature": 0.3,
        "enable_thinking": True,
    }
    started = time.monotonic()
    body = post_json(base_url, "/v1/chat/completions", payload, timeout=1800)
    elapsed = time.monotonic() - started
    usage = body.get("usage") or {}
    choice = (body.get("choices") or [{}])[0]
    message = choice.get("message", {})
    return {
        "latency_s": elapsed,
        "completion_tokens": usage.get("completion_tokens"),
        "reasoning_chars": len(message.get("reasoning_content") or ""),
        "answer_chars": len(message.get("content") or ""),
    }


def _fmt(value: Optional[float], digits: int = 1) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def run(args: argparse.Namespace) -> Dict[str, Any]:
    gpu_name = serve_metrics.read_gpu(args.gpu_index)
    report: Dict[str, Any] = {
        "meta": {
            "base_url": args.base_url,
            "model": args.model,
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
            "gpu_present": gpu_name is not None,
        },
        "warm_ttft": [],
        "decode": [],
        "thinking": [],
    }

    sampler = serve_metrics.MetricsSampler(interval_s=args.metrics_interval, gpu_index=args.gpu_index)
    sampler.start()

    prefix_messages: List[Dict[str, str]] = [
        {"role": "system", "content": SYSTEM_PREFIX},
        {"role": "user", "content": "Acknowledge the operational context in one sentence."},
    ]
    # The first request prewarms the shared prefix; its TTFT doubles as the cold number.
    cold = measure_ttft_ms(args.base_url, args.model, prefix_messages, max_tokens=16)
    report["cold"] = cold
    print(f"cold first-request ttft: {_fmt(cold['ttft_ms'], 0)} ms (prefix prewarm)")

    prefix_tokens = estimated_tokens(SYSTEM_PREFIX)
    print(f"warm ttft (prefix reuse, {args.trials} trials each):")
    for size in args.prompt_tokens:
        user = (
            "Context recap and queue for this shift:\n"
            + filler_to_target_tokens(size, prefix_tokens)
            + "\nUsing the above, answer in one short sentence: is the evening settlement "
            "reconciliation on schedule?"
        )
        messages = [
            {"role": "system", "content": SYSTEM_PREFIX},
            {"role": "user", "content": user},
        ]
        trials = [
            measure_ttft_ms(args.base_url, args.model, messages, max_tokens=16)
            for _ in range(args.trials)
        ]
        ttfts = sorted(t["ttft_ms"] for t in trials if t["ttft_ms"] is not None)
        entry = {
            "prompt_tokens_est": prefix_tokens + estimated_tokens(user),
            "trials": trials,
            "ttft_ms_mean": (sum(ttfts) / len(ttfts)) if ttfts else None,
            "ttft_ms_max": ttfts[-1] if ttfts else None,
        }
        report["warm_ttft"].append(entry)
        print(
            f"  prompt ~{entry['prompt_tokens_est']:6d} tok: mean {_fmt(entry['ttft_ms_mean'], 0)} ms"
            f"  max {_fmt(entry['ttft_ms_max'], 0)} ms"
        )

    print("decode throughput (non-thinking):")
    for concurrency in args.concurrency:
        result = decode_throughput(args.base_url, args.model, concurrency, args.decode_tokens, prefix_messages)
        report["decode"].append(result)
        print(
            f"  c={concurrency}: {concurrency * args.decode_tokens} tok in {_fmt(result['wall_s'], 2)}s"
            f" -> {_fmt(result['aggregate_tok_s'])} tok/s aggregate"
            f" ({_fmt(result['per_stream_tok_s_mean'])} per stream)"
        )

    print("thinking tasks:")
    for index, task in enumerate(THINKING_TASKS[: args.thinking_tasks], start=1):
        result = thinking_task(args.base_url, args.model, task, args.thinking_cap, prefix_messages)
        result["task"] = task
        report["thinking"].append(result)
        print(
            f"  task {index}: {_fmt(result['latency_s'], 2)}s end-to-end,"
            f" {result['completion_tokens']} completion tokens"
        )

    summary = sampler.stop()
    report["metrics"] = summary.as_dict()
    print(summary.format())

    if args.metrics_csv:
        sampler.write_csv(args.metrics_csv)
        print(f"metrics time-series written to {args.metrics_csv}")
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report written to {args.out}")
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080", help="serving base URL")
    parser.add_argument("--model", default="qwen3.8-27b", help="model id the server publishes")
    parser.add_argument("--prompt-tokens", nargs="+", type=int, default=[512, 2048, 7168],
                        help="approximate warm prompt sizes to probe (default 512 2048 7168)")
    parser.add_argument("--trials", type=int, default=3, help="TTFT trials per prompt size")
    parser.add_argument("--concurrency", nargs="+", type=int, default=[1, 8],
                        help="decode concurrency levels (default 1 8)")
    parser.add_argument("--decode-tokens", type=int, default=512)
    parser.add_argument("--thinking-tasks", type=int, default=2, help="0 skips the thinking block")
    parser.add_argument("--thinking-cap", type=int, default=2048, help="max tokens per thinking task")
    parser.add_argument("--metrics-interval", type=float, default=0.5)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--out", type=Path, help="JSON report path")
    parser.add_argument("--metrics-csv", type=Path, help="metrics time-series CSV path")
    args = parser.parse_args(argv)

    report = run(args)
    worst = max(
        (entry["ttft_ms_mean"] for entry in report["warm_ttft"] if entry["ttft_ms_mean"] is not None),
        default=None,
    )
    if worst is not None:
        verdict = "within" if worst <= 500 else "above"
        print(f"sub-500ms warm TTFT target: worst mean {worst:.0f} ms -> {verdict} target")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
