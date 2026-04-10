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

| Folder | Model | Parameters | VRAM (BF16) | Fits H100 80GB |
|--------|-------|-----------|-------------|----------------|
| `gemma-4-31b-it` | google/gemma-4-31b-it | 30.7B (dense) | ~62 GB | Yes |

## Build

```bash
docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest models/gemma-4-31b-it/
docker push ghcr.io/privasys/confidential-ai-gemma4:latest
```

## Deploy

Model weights are downloaded at runtime from HuggingFace. For gated models, set
`HF_TOKEN` as a container environment variable.

## Reproducibility

All model configs use vLLM V0 engine with `--enforce-eager` and `--seed 0` for
deterministic output on the same hardware. See the main README for details on
the reproducibility protocol.
