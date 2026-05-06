# Model Configurations

Per-model Dockerfiles and configs for confidential AI inference. Each folder
contains a standalone Dockerfile based on the official `vllm/vllm-openai` image,
configured for reproducible inference inside TDX Confidential VMs.

## Recommended Models for H100 80GB (TDX CC Mode)

| Model | Parameters | Precision | VRAM | Best Use Case |
|-------|-----------|-----------|------|---------------|
| google/gemma-4-31b-it | 30.7B (dense) | BF16 | ~62 GB | General reasoning, multimodal |
| mistralai/Mistral-Small-4 | ~30B | FP8 | ~35 GB | High-throughput production |
| Qwen/Qwen3.6-Plus-32B | 32B | BF16 | ~64 GB | Coding and agentic workflows |
| meta-llama/Llama-4-Scout-109B | 109B MoE (17B active) | INT4 | ~58 GB | Complex logic, long context |

Note: CC mode reduces usable VRAM to ~78.7 GiB (firmware overhead). BF16 and FP8
are preferred for CC because higher-precision weights are simpler to attest and verify.

## Generic Image (Recommended)

Use `Dockerfile.prod` with `MODEL_NAME` env var for any HuggingFace model:

```bash
docker run --gpus all -p 8000:8000 -p 8080:8080 \
  -e MODEL_NAME=google/gemma-4-31b-it \
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

The current convention is a short slug, e.g. `gemma4-31b`, **not** a
filesystem path like `/models/gemma4-31b`. A path-style id used to leak
through when the public fleet was hand-seeded; the chat UI then sent
`/models/gemma4-31b` while vLLM was serving `gemma4-31b`, producing a
404 with "The model /models/gemma4-31b does not exist". Always pick a
short slug here and re-use it everywhere.

### Loading a model

There is **no `management-service` command that loads a model.** Model
loading is a confidential-ai proxy operation:

```bash
curl -X POST -H "Authorization: Bearer $LOAD_TOKEN" \
     -H 'Content-Type: application/json' \
     https://<enclave-host>/v1/models/load \
     -d '{"model":"gemma4-31b"}'
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
