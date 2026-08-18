# NInfer on RunPod Serverless

One RunPod serverless worker = one resident `ninfer-serve` process. The Python
handler boots the engine when the worker starts (not on the first job) and keeps
it alive for the worker's lifetime, so warm requests get the full engine: paged
KV cache, prefix reuse (the sub-500 ms TTFT lever), CUDA graphs, and the 1-8
request decode batch. Billing is per-second while a worker is active, and the
endpoint scales to zero when idle.

This lane deploys the two groupwise-int (4-bit) artifacts at an 8K context cap
and measures them: `qwen3.8-27b` (dense) and `qwen3.6-35b-a3b` (3B active).
Both run on H100 (`sm_90a`); the nvfp4 variants are Blackwell-only and are not
deployable here.

## Layout

| File | Role |
|---|---|
| `worker.py` | serverless handler: resident engine sidecar, proxy, warmup, `[metrics]` logging |
| `Dockerfile.serverless` | H100 image: engine built for `sm_90a` + handler |
| `bridge.py` | local OpenAI-compatible HTTP bridge to the serverless endpoint (SSE streaming) |
| `tools/bench/run_serve_warm_probe.py` | warm TTFT / decode tok/s / thinking latency + GPU/CPU metrics |
| `tools/bench/run_serve_convo.py` | OpenAI-client multi-turn streaming conversation check |
| `tools/bench/serve_metrics.py` | shared GPU/CPU sampler (nvidia-smi + /proc/stat) |

## Build and push

From the repository root (the image builds the engine for `sm_90a`, H100-only):

```bash
docker build -f deploy/runpod/Dockerfile.serverless \
    --tag ghcr.io/<owner>/ninfer-h100-serverless:latest .
docker push ghcr.io/<owner>/ninfer-h100-serverless:latest
```

## Endpoint configuration

| Setting | Value | Why |
|---|---|---|
| GPU | NVIDIA H100 80GB HBM3 (SXM) | the port target |
| Container image | the one above | sm_90a build + worker handler |
| Network volume | mount at `/models` with both groupwise-int artifacts | keeps the image small; volume read is part of cold start |
| Active workers | 0 (scale to zero) | the cost goal; raise to 1 during business hours if cold starts hurt |
| Max workers | 1 per endpoint (raise later) | one H100 = one engine; add workers/endpoints to scale parallelism |
| Idle timeout | 300-1800 s | longer keeps prefix-cache warmth across quiet gaps; costs active GPU time |

Create one endpoint per model so comparisons are isolated: endpoint A with
`NINFER_MODEL=/models/qwen3_8_27b.ninfer`, endpoint B with
`NINFER_MODEL=/models/qwen3_6_35b_a3b.ninfer`. Fetch the artifacts once onto the
network volume (`hf download neroued/Qwen3.8-27B-NInfer qwen3_8_27b.ninfer
--local-dir models` and `hf download neroued/Qwen3.6-35B-A3B-NInfer
qwen3_6_35b_a3b.ninfer --local-dir models`).

Worker environment variables (all optional):

| Variable | Default | Meaning |
|---|---|---|
| `NINFER_MODEL` | `/models/qwen3_8_27b.ninfer` | artifact path; one endpoint per model |
| `NINFER_MAX_CONTEXT` | `8192` | per-request context cap |
| `NINFER_MAX_CONCURRENCY` | `8` | engine request slots |
| `NINFER_SPEC` / `NINFER_DRAFT_TOKENS` | `mtp` / `3` | speculative decoding |
| `NINFER_PRELOAD` | `1` | load the engine at worker boot, not on first job |
| `NINFER_METRICS_INTERVAL_S` | `30` | `[metrics] cpu/gpu/vram/power/temp` lines in worker logs; `0` disables |
| `NINFER_KV_CAPACITY`, `NINFER_MODEL_ID` | unset | pass through to ninfer-serve |
| `NINFER_SERVE_ARGS` | empty | extra raw flags, e.g. `--prefill-chunk 1024 --kv-dtype int8` |
| `NINFER_LOAD_TIMEOUT_S` | `900` | cold-start budget for volume load + warmup |

## Request contract

Job input is the OpenAI chat-completions body the engine already serves
(`docs/serving.md`), so clairvoyance keeps its OpenAI client shape:

```python
import runpod
runpod.api_key = "..."
endpoint = runpod.Endpoint("<endpoint-id>")
job = endpoint.run({
    "messages": [{"role": "user", "content": "..."}],
    "max_tokens": 512,
    "enable_thinking": False,     # or True
})
print(job.output())               # or: for chunk in job.stream(): ...
```

`{"route": ..., "payload": ...}` selects another engine route (the bridge uses
this). `"stream": true` streams SSE chunks back as job output parts.

## Benchmark and observe

Direct engine numbers (cleanest): run `ninfer-serve` on the GPU host and probe
it there so the GPU/CPU sampler sees the real host. Through the serverless
lane, point the same tools at the bridge.

```bash
# 1. warm TTFT / decode tok/s / thinking tasks + GPU/CPU summary (GPU host)
python3 -m tools.bench.run_serve_warm_probe \
    --base-url http://127.0.0.1:8080 --model qwen3.8-27b \
    --out probe_27b.json --metrics-csv probe_27b.csv
# repeat with --model qwen3.6-35b-a3b for the second artifact

# 2. serverless path: bridge + the same probe / the conversation check
export RUNPOD_API_KEY=... RUNPOD_ENDPOINT_ID=<id>
python3 deploy/runpod/bridge.py --port 8081 --model-name qwen3.8-27b &
python3 -m tools.bench.run_serve_warm_probe --base-url http://127.0.0.1:8081 \
    --model qwen3.8-27b --out probe_serverless_27b.json

# 3. how an actual conversation feels, OpenAI client, streamed turns
pip install openai
python3 -m tools.bench.run_serve_convo \
    --base-url http://127.0.0.1:8081/v1 --model qwen3.8-27b --turns 6
python3 -m tools.bench.run_serve_convo \
    --base-url http://127.0.0.1:8081/v1 --model qwen3.8-27b --turns 6 --thinking
```

The warm probe reports cold first-request TTFT, warm TTFT per prompt size
(prefix reuse on), aggregate decode tok/s at C=1/8, thinking-task latency, and
a GPU/CPU summary (util/vram/power/temp sampled for the whole run, time-series
in the CSV). On serverless, GPU/CPU metrics also stream into the RunPod worker
logs as `[metrics]` lines.

## Clairvoyance integration

Clairvoyance keeps using the OpenAI adapter unchanged; only `base_url` changes:

```python
from openai import OpenAI
client = OpenAI(base_url="http://<bridge-host>:8081/v1", api_key="-")
stream = client.chat.completions.create(
    model="qwen3.8-27b",
    messages=[...],
    stream=True,
    extra_body={"enable_thinking": True},
)
```

Run the bridge next to clairvoyance (same host or a small always-on box) and
point it at the model endpoint you want. Direct serving (`ninfer-serve` on a
pod or local GPU) works with the same client and no bridge.

## What to expect

| Path | Latency |
|---|---|
| Warm request | engine TTFT (~50-300 ms) + RunPod dispatch (~100-300 ms) ≈ 200-600 ms to first token |
| Cold request (first after idle) | + 30-90 s volume load + graph capture — one-time per idle period |
| India client | + ~150-250 ms RTT; RunPod has no India region, so the worker sits US/EU |

Notes that matter:

- Sub-500 ms TTFT is realistic only on the warm path with prefix reuse (default
  on). A cold start is a one-time 30-90 s hit per idle period, not per request.
- RunPod dispatches jobs serially to a single-worker handler by default. The
  engine still keeps state warm between jobs; for parallel throughput raise max
  workers (each gets its own engine) rather than relying on intra-worker batch.
- A scheduled keep-warm ping from clairvoyance (one tiny generation every few
  minutes during business hours) holds one worker active, converting cold starts
  into warm ones for the hours you actually pay for.
- Flash (`runpod-flash`) is fine for iterating handler logic locally with
  `flash dev`, but deploy the standard RunPod serverless endpoint with this
  custom image: the engine binary requires the sm_90a CUDA build that Flash's
  default packaging does not carry.

## Cost shape (why serverless here)

At H100 ≈ $2.8-3.3/hr per active worker: paying only for active time beats a
24/7 pod (~$2,000-2,400/mo) below roughly 25-30% GPU utilization. A clairvoyance
trial doing a few hours of GPU-active work per day lands around $150-250/mo plus
~$4-8/mo for the network volume. If steady utilization climbs past a quarter of
the day, a pod becomes cheaper again.
