# confidential-ai

Attested, reproducible LLM inference for confidential virtual machines (Intel TDX with an NVIDIA H100 in confidential-computing mode).

A Go proxy in front of [vLLM](https://github.com/vllm-project/vllm) that serves one chat model and, optionally, an embedding model and a reranker on the same GPU. It speaks both the **OpenAI** (`/v1/chat/completions`, `/v1/completions`, `/v1/embeddings`) and the **Anthropic Messages** (`/v1/messages`) wires, authenticates every caller, meters usage per caller, and can return a reproducibility record that lets anyone replay a response byte for byte on the same image and hardware.

## Architecture

```
caller ──RA-TLS──> confidential-ai proxy ($PORT)
                     ├─> vLLM generate  :8000   chat / messages / completions
                     ├─> vLLM embed     :8001   embeddings   (optional)
                     └─> vLLM rerank    :8002   rerank       (optional)
                                    └── one H100, GPU memory split between them
```

vLLM listens on loopback only; every request goes through the proxy. The proxy:

1. Identifies the caller (Privasys ID token, sealed-transport subject, or an attested peer app) and checks it has a funded account.
2. On the generate path, pins the sampling: a random `seed` when none is sent, the model's own `temperature`/`top_p`/`top_k` when absent, a UTC clock injected into the last user turn, and a per-caller `cache_salt` so the prefix cache is never shared across callers.
3. Forwards to the right vLLM instance and meters prompt and completion tokens.
4. With `X-Privasys-Reproducibility`, attaches the reproducibility record to the response.

## API

| Route | Purpose | Auth |
|---|---|---|
| `POST /v1/chat/completions` | OpenAI chat completions, streaming and tools | caller |
| `POST /v1/messages` | Anthropic Messages, streaming and tools | caller |
| `POST /v1/messages/count_tokens` | Anthropic token count (not metered) | caller |
| `POST /v1/completions` | OpenAI text completions | caller |
| `POST /v1/embeddings` | Embeddings from the embed model | caller |
| `POST /v1/rerank` | Reranking from the rerank model | caller |
| `GET /v1/models` | Served models (vLLM listing) | none |
| `POST /v1/models/load`, `/v1/models/unload` | Declare or drop the model set | app owner or admin |
| `GET /v1/models/status` | Load state and progress per model | none |
| `GET /readiness` | 200 only when every configured model is up | none |
| `GET /health`, `/healthz` | Build, GPU, TEE and per-model state | none |
| `GET /.well-known/attestation-extensions` | X.509 extensions the platform stamps into the certificate | none |
| `GET /.well-known/served-models` | Attested serving manifest: image, vLLM, and each model's digest | none |
| `POST /configure` | Billing account and usage-report endpoint | app owner or admin, or the platform |
| `GET /metrics` | Prometheus counters | none |

Chat, Messages, completions and `/v1/models` answer as soon as the chat model is up; embeddings and rerank wait for their own model. `/readiness` stays 503 until all of them are loaded.

**Callers** reach the proxy only through the platform: attested apps over mutual RA-TLS (identified by app id), and people over the sealed transport (identified by their Privasys ID subject), which is also how `privasys apps call` reaches it. The proxy also accepts a Privasys ID access token as `Authorization: Bearer` or `x-api-key`, the headers OpenAI and Anthropic clients send. A caller without a funded Privasys account gets `402`.

## Reproducibility

Send `X-Privasys-Reproducibility: 1` (or `strict`, which also gives the request a single-use prefix-cache salt) and the response carries:

```json
{
  "reproducibility": {
    "request_id": "…",
    "seed": 2863311530,
    "temperature": 0.7, "top_p": 0.8, "top_k": 20, "max_tokens": 1024,
    "model": "qwen36-35b-a3b-fp8",
    "quantization": "fp8",
    "vllm_version": "0.29.0",
    "cuda_version": "12.6.3",
    "gpu": "H100-80GB",
    "tensor_parallel_size": 1,
    "batch_invariance": true,
    "image_digest": "sha256:…",
    "tee_type": "tdx",
    "timestamp": "2026-11-19T12:00:00Z",
    "dynamic_context": "…",
    "kv_cache_mode": "session",
    "cached_tokens": 0
  }
}
```

On chat completions it is a top-level key (or one `data:` frame before `[DONE]` when streaming). On the Messages wire it is an `event: privasys_reproducibility` just before `message_stop`, so typed-event clients accept it. To replay, send the same request with the recorded `seed` and the recorded `dynamic_context` in `X-Privasys-Dynamic-Context`; vLLM runs with `VLLM_BATCH_INVARIANT=1` and patches that keep the hybrid-attention kernels batch-invariant, so the output does not depend on what else the GPU was serving.

## Models

`POST /v1/models/load` declares the whole set in one call (`model`, optional `embedding_model` and `rerank_model`, plus per-model vLLM settings such as `max_model_len`, `kv_cache_dtype`, `max_num_seqs`). A model is a Hugging Face repository id, downloaded once into the app's encrypted volume, or a directory under `/models`. The set is saved on the volume and reloaded at every boot. Weights are read with vLLM's multi-threaded loader (`load_threads`, default 8).

Each model's digest covers every byte vLLM reads. For a model on a dm-verity disk it is the disk's root hash. Otherwise it is the SHA-256 of a manifest in `sha256sum` format, one line per file sorted by path, which anyone can recompute from the same files:

```bash
cd <snapshot> && find -L . -type f ! -name '.*' | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum
```

A Hugging Face weight file is stored under its SHA-256; its content is checked against that name the first time it is used, and a file that does not match fails the load. `/.well-known/served-models` lists each model with its digest and the manifest's files, so a client can check every file against the hash the publisher lists; the chat model's digest is also stamped into the certificate as OID `1.3.6.1.4.1.65230.5.4.5`.

## Building

```bash
go test ./...
docker build -f Dockerfile.prod -t confidential-ai .
```

The image (vLLM 0.29 with CUDA 12.6, the proxy, and the batch-invariance patches in `patches/vllm/`) is built by CI on every push to `main` and published as `ghcr.io/privasys/confidential-ai`. It runs as an app on [Enclave OS Virtual](https://github.com/Privasys/enclave-os-virtual), which provides the attested TLS, the encrypted volume and the GPU in confidential-computing mode.

Documentation: [docs.privasys.org/technology/confidential-ai](https://docs.privasys.org/technology/confidential-ai/overview).

## License

GNU Affero General Public License v3.0. See [LICENSE](LICENSE).
