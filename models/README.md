# Model Configurations

Per-model Dockerfiles and configs for confidential AI inference. Each folder
contains a standalone Dockerfile based on the official `vllm/vllm-openai` image,
configured for reproducible inference inside TDX Confidential VMs.

## Models

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
