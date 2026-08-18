#!/usr/bin/env python3
"""Multi-turn streaming conversation check using the official OpenAI client.

This is the clairvoyance-integration test: point the OpenAI client's base_url
at a running ninfer-serve (or the RunPod bridge, deploy/runpod/bridge.py) and
drive an actual payments-ops conversation with streamed turns, thinking on or
off. Per-turn TTFT, inter-token latency, and throughput are reported; with
--live the assistant text streams to the terminal so the conversation feel is
directly visible.

  pip install openai
  python3 -m tools.bench.run_serve_convo \
      --base-url http://127.0.0.1:8080/v1 --model qwen3.8-27b --turns 6
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.bench.serve_metrics import percentile  # noqa: E402

SYSTEM_PROMPT = (
    "You are the clairvoyance payments assistant for a large Indian payment "
    "processor. You help operations engineers with UPI, card settlements, "
    "chargebacks, refunds, and fraud queues. Be concise and precise."
)
CONVERSATION_TURNS = (
    "What's our refund policy for amounts above rupees five lakh?",
    "A merchant says a UPI payment is stuck at pending for 25 minutes. Is that normal?",
    "Now the same merchant reports the transaction failed but the customer was debited. "
    "What are the next steps?",
    "How long do we have to respond to a domestic card chargeback, and what evidence do we attach?",
    "If evening UPI volume doubles during a partner-bank maintenance window, what should we monitor first?",
    "Draft a two-line status note we could publish if pending transactions spike past thirty minutes.",
    "Summarize today's incident risks from this conversation in three bullet points.",
    "Which of these needed human judgment rather than the runbook?",
)

# Inter-token latency bands for the "how it feels" verdict, in milliseconds.
ITL_INSTANT_MS = 40.0
ITL_FLUID_MS = 80.0
ITL_CHUNKY_MS = 150.0


def run_turn(client: Any, model: str, messages: List[Dict[str, Any]], args: argparse.Namespace) -> Dict[str, Any]:
    request = {
        "model": model,
        "messages": messages,
        "max_tokens": args.max_tokens,
        "temperature": 0.3,
        "stream": True,
        # ninfer-serve accepts enable_thinking per request (docs/serving.md).
        "enable_thinking": args.thinking,
    }
    if args.reasoning_effort:
        request["reasoning_effort"] = args.reasoning_effort

    started = time.monotonic()
    ttft_s: Optional[float] = None
    token_times: List[float] = []
    reasoning_chars = 0
    answer_chars = 0
    answer_parts: List[str] = []
    use_ansi = args.live and sys.stdout.isatty()

    stream = client.chat.completions.create(**request)
    for chunk in stream:
        if not getattr(chunk, "choices", None):
            continue
        delta = chunk.choices[0].delta or {}
        reasoning = getattr(delta, "reasoning_content", None)
        content = getattr(delta, "content", None)
        emitted = bool(reasoning) or bool(content)
        if reasoning:
            reasoning_chars += len(reasoning)
            if args.live:
                text = f"\033[2m{reasoning}\033[0m" if use_ansi else reasoning
                sys.stdout.write(text)
                sys.stdout.flush()
        if content:
            answer_chars += len(content)
            answer_parts.append(content)
            token_times.append(time.monotonic())
            if args.live:
                sys.stdout.write(content)
                sys.stdout.flush()
        if emitted and ttft_s is None:
            ttft_s = time.monotonic() - started
    total_s = time.monotonic() - started

    inter_token_ms = [
        (token_times[i] - token_times[i - 1]) * 1000.0 for i in range(1, len(token_times))
    ]
    if args.live:
        sys.stdout.write("\n\n")
        sys.stdout.flush()
    return {
        "ttft_ms": ttft_s * 1000.0 if ttft_s is not None else None,
        "total_s": total_s,
        "tokens_streamed": len(token_times),
        "tok_s": len(token_times) / total_s if total_s > 0 and token_times else None,
        "itl_ms_p50": percentile(sorted(inter_token_ms), 0.50) if inter_token_ms else None,
        "itl_ms_p95": percentile(sorted(inter_token_ms), 0.95) if inter_token_ms else None,
        "itl_ms_max": max(inter_token_ms) if inter_token_ms else None,
        "reasoning_chars": reasoning_chars,
        "answer_chars": answer_chars,
        "answer": "".join(answer_parts),
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", default="http://127.0.0.1:8080/v1",
                        help="OpenAI-compatible base URL (ninfer-serve or the RunPod bridge)")
    parser.add_argument("--model", default="qwen3.8-27b")
    parser.add_argument("--api-key", default="ninfer", help="ignored by ninfer-serve; any non-empty string")
    parser.add_argument("--turns", type=int, default=6, help="conversation turns to run")
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument("--thinking", action="store_true", help="enable thinking for every turn")
    parser.add_argument("--reasoning-effort", default=None,
                        help="optional reasoning_effort (e.g. low) alongside thinking")
    parser.add_argument("--no-live", dest="live", action="store_false",
                        help="do not stream text to the terminal")
    parser.add_argument("--json-out", type=Path, help="write per-turn stats as JSON")
    args = parser.parse_args(argv)

    try:
        from openai import OpenAI
    except ImportError:
        print("error: the openai package is required (pip install openai)", file=sys.stderr)
        return 2

    if args.turns <= 0:
        print("no turns requested; nothing to do")
        return 0

    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    try:
        server_models = [m.id for m in client.models.list()]
    except Exception as error:
        server_models = []
        print(f"note: could not list server models ({error}); using '{args.model}' as-is")
    if server_models and args.model not in server_models:
        print(f"note: model '{args.model}' not in server list {server_models}; using {server_models[0]}")
        args.model = server_models[0]

    messages: List[Dict[str, Any]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    turns = CONVERSATION_TURNS[: args.turns]
    results: List[Dict[str, Any]] = []

    print(f"conversation: model={args.model} thinking={'on' if args.thinking else 'off'} "
          f"turns={len(turns)} live={args.live}\n")
    for index, user_text in enumerate(turns, start=1):
        if args.live:
            print(f"[user {index}] {user_text}")
            print(f"[assistant {index}] ", end="", flush=True)
        messages.append({"role": "user", "content": user_text})
        result = run_turn(client, args.model, messages, args)
        result["turn"] = index
        result["user"] = user_text
        results.append(result)
        # Keep the conversation structurally valid even if the engine emitted
        # nothing (an empty assistant turn can 400 the next request).
        messages.append({"role": "assistant", "content": result["answer"] or "(no content)"})

    def cell(value: Optional[float], spec: str) -> str:
        return format(value, spec) if value is not None else "n/a"

    ttfts = sorted(r["ttft_ms"] for r in results if r["ttft_ms"] is not None)
    median_itls = sorted(r["itl_ms_p50"] for r in results if r["itl_ms_p50"] is not None)
    all_tok_s = [r["tok_s"] for r in results if r["tok_s"]]
    print("\nper-turn summary")
    header = ("turn", "ttft_ms", "tokens", "tok/s", "itl_p50_ms", "itl_p95_ms")
    print(f"{header[0]:>4} {header[1]:>8} {header[2]:>7} {header[3]:>7} {header[4]:>11} {header[5]:>11}")
    for r in results:
        print(
            f"{r['turn']:>4} "
            f"{cell(r['ttft_ms'], '>8.0f')} "
            f"{r['tokens_streamed']:>7} "
            f"{cell(r['tok_s'], '>7.1f')} "
            f"{cell(r['itl_ms_p50'], '>11.1f')} "
            f"{cell(r['itl_ms_p95'], '>11.1f')}"
        )

    if ttfts:
        print(f"\nttft: mean {sum(ttfts)/len(ttfts):.0f} ms  max {ttfts[-1]:.0f} ms")
    if all_tok_s:
        print(f"decode: mean {sum(all_tok_s)/len(all_tok_s):.1f} tok/s across turns")
    if median_itls:
        median_itl = median_itls[len(median_itls) // 2]
        if median_itl <= ITL_INSTANT_MS:
            feel = "reads as instant"
        elif median_itl <= ITL_FLUID_MS:
            feel = "reads as fluid typing"
        elif median_itl <= ITL_CHUNKY_MS:
            feel = "noticeably chunky"
        else:
            feel = "feels sluggish"
        print(f"conversation feel: median inter-token latency {median_itl:.1f} ms -> {feel}")

    if args.json_out:
        args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
        print(f"per-turn stats written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
