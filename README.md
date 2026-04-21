# confidential-ai

Reproducible LLM inference proxy for confidential virtual machines (Intel TDX + NVIDIA H100).

Wraps [vLLM](https://github.com/vllm-project/vllm) and injects reproducibility metadata into every response, enabling fully verifiable AI inference inside hardware-attested environments.

## Architecture

```
Client
  |
  +--> confidential-ai proxy (Go, port 8080)
         |
         +--> vLLM backend (Python, port 8000)
                |
                +--> H100 80GB
```

The proxy:
1. Receives OpenAI-compatible requests
2. Injects `seed` if not present (default: 0)
3. Forwards to vLLM
4. Wraps the response with `reproducibility` metadata block

## Reproducibility Metadata

Every inference response includes:

```json
{
  "reproducibility": {
    "request_id": "uuid",
    "seed": 0,
    "temperature": 0.7,
    "top_p": 0.95,
    "model": "google/gemma-4-31b-it",
    "quantization": "",
    "vllm_version": "0.19.1",
    "cuda_version": "13.0",
    "gpu": "H100-80GB",
    "tensor_parallel_size": 1,
    "batch_invariance": true,
    "image_digest": "sha256:...",
    "tee_type": "tdx",
    "timestamp": "2026-04-06T12:00:00Z"
  }
}
```

Any verifier can replay the exact same request on the same image+hardware
and compare output token-by-token.

## Running

### Standalone (proxy only, expects vLLM running)

```bash
go build -o confidential-ai ./cmd/server/
./confidential-ai \
  --listen :8080 \
  --vllm-upstream http://localhost:8000 \
  --model google/gemma-4-31b-it \
  --gpu-type H100-80GB \
  --tee-type tdx \
  --image-digest sha256:abc123
```

### Docker (full stack: vLLM + proxy)

```bash
docker build -f Dockerfile.prod -t confidential-ai .
docker run --gpus all -p 8080:8080 -p 8000:8000 \
  -e MODEL_NAME=google/gemma-4-31b-it \
  -e HF_TOKEN=<your-token> \
  -v model-cache:/root/.cache/huggingface \
  confidential-ai
```

See [models/](models/) for pre-configured model images and the full list
of supported models.

## Pre-built Model Images

Each model image extends `ghcr.io/privasys/confidential-ai:latest` and declares
its GCP persistent disk via the `ai.privasys.volume` OCI label. The manager
reads this label at container start and bind-mounts the disk into the container.

| Image | Model | Parameters | Precision | Disk |
|-------|-------|-----------|-----------|------|
| `confidential-ai-gemma4` | google/gemma-4-31b-it | 30.7B (dense) | BF16 | model-gemma4-31b (70 GB) |
| `confidential-ai-qwen25` | Qwen/Qwen2.5-32B-Instruct | 32.5B (dense) | BF16 | model-qwen25-32b (75 GB) |
| `confidential-ai-mistral-small` | mistralai/Mistral-Small-24B-Instruct-2501 | 24B (dense) | BF16 | model-mistral-small-24b (55 GB) |
| `confidential-ai-llama4-scout` | meta-llama/Llama-4-Scout-17B-16E-Instruct | 109B MoE (17B active) | BF16 + INT4 | model-llama4-scout (240 GB) |

All images are published to `ghcr.io/privasys/` and built automatically by CI
on every push to `main`.

### GCP Confidential VM (a3-highgpu-1g, TDX)

See `.operations/confidential-ai.md` for full deployment plan and `deploy/` for scripts.

## API

### POST /v1/chat/completions

OpenAI-compatible chat completions with reproducibility metadata.

### POST /v1/completions

OpenAI-compatible text completions with reproducibility metadata.

### GET /v1/models

Proxied directly from vLLM.

### GET /health

Returns server status and config metadata as JSON.

### GET /metrics

Prometheus-compatible metrics.

## Testing

```bash
go test ./...
```

## License

See top-level [LICENSE](../../LICENSE).
