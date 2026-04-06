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
                +--> H100 80GB (INT4 AWQ quantization)
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
    "model": "gpt-oss-120b",
    "quantization": "awq",
    "vllm_version": "0.19.0",
    "cuda_version": "12.6",
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
  --model gpt-oss-120b \
  --quantization awq \
  --gpu-type H100-80GB \
  --tee-type tdx \
  --image-digest sha256:abc123
```

### Docker (full stack: vLLM + proxy)

```bash
docker build -t confidential-ai .
docker run --gpus all -p 8080:8080 \
  -e MODEL_NAME=gpt-oss-120b \
  -e QUANTIZATION=awq \
  confidential-ai
```

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
