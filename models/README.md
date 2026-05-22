# Model Configurations

Per-model Dockerfiles and configs for confidential AI inference. Each folder
contains a standalone Dockerfile based on the official `vllm/vllm-openai` image,
configured for reproducible inference inside TDX Confidential VMs.

## Recommended Models for H100 80GB (TDX CC Mode)

| Model | Parameters | Precision | VRAM | Best Use Case |
|-------|-----------|-----------|------|---------------|
| Qwen/Qwen3.6-35B-A3B-FP8 | 35B (MoE, ~3B active) | FP8 | ~60 GB | Default — coding, agentic, tool use |
| google/gemma-4-31b-it | 30.7B (dense) | BF16 | ~62 GB | General reasoning, multimodal |
| mistralai/Mistral-Small-24B-Instruct-2501 | 24B (dense) | BF16 | ~48 GB | High-throughput production |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | 109B MoE (17B active) | BF16 + INT4 | ~58 GB | Complex logic, long context |

Note: CC mode reduces usable VRAM to ~78.7 GiB (firmware overhead). BF16 and FP8
are preferred for CC because higher-precision weights are simpler to attest and verify.

## Generic Image (Recommended)

Use `Dockerfile.prod` with `MODEL_NAME` env var for any HuggingFace model:

```bash
docker run --gpus all -p 8000:8000 -p 8080:8080 \
  -e MODEL_NAME=Qwen/Qwen3.6-35B-A3B-FP8 \
  -e HF_TOKEN=<token> \
  -v model-cache:/root/.cache/huggingface \
  ghcr.io/privasys/confidential-ai:latest
```

Optional env vars: `DTYPE` (default: auto), `QUANTIZATION` (awq/gptq/fp8),
`MAX_MODEL_LEN` (default: 8192), `GPU_MEMORY_UTILIZATION` (default: 0.90).

## Per-Model Images

| Folder | Model | Parameters | Precision | VRAM | Disk |
|--------|-------|-----------|-----------|------|------|
| `gemma-4-31b-it` | google/gemma-4-31b-it | 30.7B (dense) | BF16 | ~62 GB | model-gemma4-31b (70 GB) |
| `qwen2.5-32b-instruct` | Qwen/Qwen2.5-32B-Instruct | 32.5B (dense) | BF16 | ~65 GB | model-qwen25-32b (75 GB) |
| `mistral-small-24b-instruct-2501` | mistralai/Mistral-Small-24B-Instruct-2501 | 24B (dense) | BF16 | ~48 GB | model-mistral-small-24b (55 GB) |
| `llama-4-scout-17b-16e-instruct` | meta-llama/Llama-4-Scout-17B-16E-Instruct | 109B MoE (17B active) | BF16 + INT4 | ~58 GB | model-llama4-scout (240 GB) |

Each image declares its GCP persistent disk via a `LABEL ai.privasys.volume`
entry. The Enclave OS Virtual manager reads this label at container start and
bind-mounts the disk read-only at `/models`.

## Build

```bash
docker build -t ghcr.io/privasys/confidential-ai-qwen36:latest models/qwen3.6-35b-a3b-fp8/
docker push ghcr.io/privasys/confidential-ai-qwen36:latest
```

```bash
docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest models/gemma-4-31b-it/
docker push ghcr.io/privasys/confidential-ai-gemma4:latest
```

## Deploy

Model weights are downloaded at runtime from HuggingFace. For gated models, set
`HF_TOKEN` as a container environment variable.

## Reproducibility

vLLM is started with `--seed 0`, CUDA graphs enabled, the V1 scheduler,
`--no-enable-chunked-prefill`, and `--max-num-batched-tokens` sized to fit
any prompt in a single prefill step. Combined with the locked kernel /
NVIDIA driver / CUDA / vLLM versions and `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
this gives deterministic output for serialised requests on the same
hardware. Per-request determinism under concurrent traffic still requires
batch-invariant kernels (tracked separately). See the main README for the
full reproducibility protocol.

## Canonical model id (a.k.a. `--served-model-name`)

The string clients put in the `model` field of `POST /v1/chat/completions`
(and that `GET /v1/models` advertises) is the **canonical model id**. It
is set by the loader via vLLM's `--served-model-name` and **must equal**:

- the `name` field of every entry the management-service publishes in
  `GET /api/v1/ai/instances/<alias>` → `available_models[].name`;
- the value the chat front-end forwards verbatim (the proxy never
  rewrites `model`; see `internal/handler/handler.go` →
  `proxyWithReproducibility`).

The current convention is a short slug, e.g. `qwen36-35b-a3b-fp8`, **not** a
filesystem path like `/models/qwen36-35b-a3b-fp8`. A path-style id used to leak
through when the public fleet was hand-seeded; the chat UI then sent
`/models/qwen36-35b-a3b-fp8` while vLLM was serving `qwen36-35b-a3b-fp8`, producing a
404 with "The model /models/qwen36-35b-a3b-fp8 does not exist". Always pick a
short slug here and re-use it everywhere.

### Loading a model

There is **no `management-service` command that loads a model.** Model
loading is a confidential-ai proxy operation:

```bash
curl -X POST -H "Authorization: Bearer $LOAD_TOKEN" \
     -H 'Content-Type: application/json' \
     https://<enclave-host>/v1/models/load \
     -d '{"model":"qwen36-35b-a3b-fp8"}'
```

The state machine (`internal/models/manager.go`) goes
`idle → loading → ready` (~3 min cold). `GET /v1/models/status` returns
`{state, model, model_digest, progress, message}`. The in-image manager
(`enclave-os-virtual/internal/runtimestatus`) polls that endpoint every
30 s and pushes deltas to
`management-service/api/v1/enclave/runtime-status`. The mgmt-service then
folds `loaded_model` + `loaded_model_digest` into
`available_models[].digest` and `loaded` whenever it answers `GET
/api/v1/ai/instances/<alias>` (see `fleets.go` → `GetInstance`).

Consequence: `available_models[].digest` will be **empty** in the
instance API response whenever no enclave in the fleet currently reports
the model as `state=ready`. That is the correct behaviour — the digest
is only meaningful for a currently-loaded model, since it is the OID 3.5
verity root hash of the bytes that vLLM has actually mapped. To populate
it, ensure (a) the enclave manager has `MGMT_BASE_URL`, `ENCLAVE_TOKEN`
and `ENCLAVE_ID` set so the runtime-status push runs, and (b) the model
is loaded.

## Reasoning & tool calling

`Dockerfile.prod` pins **vLLM v0.21.0** (release notes:
[vllm-project/vllm v0.21.0](https://github.com/vllm-project/vllm/releases/tag/v0.21.0)),
which ships first-class reasoning and tool-call parsers for both Gemma 4
and Qwen3.6.

### Gemma 4

The image bakes the official chat template into
`/opt/vllm-templates/tool_chat_template_gemma4.jinja` so the manager can
reference it by short name.

When `POST /v1/models/load` receives a request whose `model` slug
contains `gemma4`, `internal/models/manager.go::doLoad` auto-applies:

```text
--enable-auto-tool-choice
--reasoning-parser gemma4
--tool-call-parser gemma4
--chat-template /opt/vllm-templates/tool_chat_template_gemma4.jinja
--default-chat-template-kwargs {"enable_thinking": true}
```

### Qwen3 / Qwen3.5 / Qwen3.6

For models matching `qwen3`, `qwen35`, or `qwen36`, the manager auto-applies:

```text
--reasoning-parser qwen3
--tool-call-parser hermes
--enable-auto-tool-choice
```

Qwen3.6 uses the Hermes tool-call schema and the `qwen3` reasoning parser
(which strips `<think>…</think>` blocks from the user-visible content).

### Shared behaviour

Callers can override any of the auto-applied flags via the new optional
`LoadRequest` fields (`reasoning_parser`, `tool_call_parser`,
`enable_auto_tool_choice`, `chat_template`, `enable_thinking`) — the
auto-defaults are only used when the corresponding field is unset.

User-visible effect: the OpenAI-compatible streaming response carries
`message.reasoning` (non-stream) or `delta.reasoning_content` (stream)
**separately** from `delta.content`. The chat front-end re-wraps the
reasoning channel in `<think>…</think>` sentinels so the existing
`splitReasoning()` / `ThinkingBlock` UI keeps working unchanged. The
system prompt no longer contains any "wrap your thoughts in `</think>`"
nudges — that was a workaround for vLLM < 0.20 and has been removed.

### SSE framing through the sealed-session relay

`internal/handler/handler.go::proxyStream` forwards the upstream vLLM
stream **one SSE event per Write+Flush** (an event is everything up to
and including the terminating blank line). The sealed-session relay in
enclave-os-virtual maps each Write to one length-prefixed AEAD-sealed
frame, so this guarantees that one sealed frame contains exactly one
SSE event. Before this change the proxy emitted the `data:` line and
the trailing blank line as two separate flushes, which doubled
AEAD/CBOR overhead and produced frames that did not align with SSE
event boundaries (so the browser SSE parser could not fire per frame).
The reproducibility metadata event is still injected immediately
before `data: [DONE]`.

### Upgrade requirement

vLLM v0.21.x requires `transformers>=5.5.0` (and formally deprecates
v4 — migrate any pinned downstream code to v5). The dependency is pulled in
transitively by the `uv pip install` in `Dockerfile.prod`. CUDA 12.6 is
still supported via `--torch-backend=cu126`, so the base image
(`nvidia/cuda:12.6.3-runtime-ubuntu24.04`) does not need to change.
