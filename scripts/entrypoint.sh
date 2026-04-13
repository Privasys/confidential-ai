#!/bin/bash
set -euo pipefail

# --- Model resolution ---------------------------------------------------
# Accepts MODEL_URL (preferred) or MODEL_NAME (legacy).
#
#   file:///models/google/gemma-4-31b-it   Local path (pre-loaded on disk)
#   hf://TOKEN@google/gemma-4-31b-it       HuggingFace with embedded token
#   hf://google/gemma-4-31b-it             HuggingFace (public or HF_TOKEN env)
#   google/gemma-4-31b-it                  Plain model name (same as hf://)
#
MODEL_URL="${MODEL_URL:-}"
MODEL_NAME="${MODEL_NAME:-}"

if [[ -n "$MODEL_URL" ]]; then
  case "$MODEL_URL" in
    file://*)
      MODEL="${MODEL_URL#file://}"
      ;;
    hf://*)
      hf_path="${MODEL_URL#hf://}"
      if [[ "$hf_path" == *@* ]]; then
        export HF_TOKEN="${hf_path%%@*}"
        MODEL="${hf_path#*@}"
      else
        MODEL="$hf_path"
      fi
      ;;
    *)
      MODEL="$MODEL_URL"
      ;;
  esac
elif [[ -n "$MODEL_NAME" ]]; then
  MODEL="$MODEL_NAME"
else
  echo "[confidential-ai] ERROR: MODEL_URL or MODEL_NAME is required"
  exit 1
fi
QUANT="${QUANTIZATION:-}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${LISTEN_ADDR:-:8080}"

echo "[confidential-ai] Starting vLLM for model=$MODEL quantization=$QUANT dtype=$DTYPE"

# Build vLLM args
VLLM_ARGS=(
  --seed 0
  --tensor-parallel-size 1
  --enforce-eager
  --no-enable-log-requests
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --dtype "$DTYPE"
  --port "$VLLM_PORT"
)

if [[ -n "$QUANT" && "$QUANT" != "none" ]]; then
  VLLM_ARGS+=(--quantization "$QUANT")
fi

# Start vLLM in the background (V0 engine for reproducibility)
VLLM_USE_V1=0 CUBLAS_WORKSPACE_CONFIG=:4096:8 PYTHONHASHSEED=0 \
vllm serve "$MODEL" "${VLLM_ARGS[@]}" &

VLLM_PID=$!

# Wait for vLLM to be ready
echo "[confidential-ai] Waiting for vLLM on port $VLLM_PORT..."
for i in $(seq 1 120); do
  if curl -sf "http://localhost:$VLLM_PORT/health" > /dev/null 2>&1; then
    echo "[confidential-ai] vLLM is ready"
    break
  fi
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[confidential-ai] vLLM process exited unexpectedly"
    exit 1
  fi
  sleep 5
done

# --- Compute model identity digest for attestation (OID 3.5) -----------
# Uses the safetensors index file (lists all weight shards + metadata)
# as a proxy for the full model identity. Falls back to config.json.
MODEL_DIGEST=""
MODEL_DIR="$MODEL"

# If MODEL is a HuggingFace repo name, resolve the cached snapshot path
if [[ ! -d "$MODEL_DIR" ]]; then
  MODEL_DIR=$(python3 -c "
from pathlib import Path
import os
cache = Path(os.environ.get('HF_HOME', Path.home() / '.cache' / 'huggingface')) / 'hub'
repo = 'models--' + '${MODEL}'.replace('/', '--')
snap_dir = cache / repo / 'snapshots'
if snap_dir.is_dir():
    snaps = sorted(snap_dir.iterdir())
    if snaps: print(snaps[-1])
" 2>/dev/null || true)
fi

if [[ -n "$MODEL_DIR" && -d "$MODEL_DIR" ]]; then
  if [[ -f "$MODEL_DIR/model.safetensors.index.json" ]]; then
    MODEL_DIGEST=$(sha256sum "$MODEL_DIR/model.safetensors.index.json" | cut -d' ' -f1)
    echo "[confidential-ai] Model digest (safetensors index): $MODEL_DIGEST"
  elif [[ -f "$MODEL_DIR/.sha256" ]]; then
    MODEL_DIGEST=$(cat "$MODEL_DIR/.sha256")
    echo "[confidential-ai] Model digest (pre-computed): $MODEL_DIGEST"
  elif [[ -f "$MODEL_DIR/config.json" ]]; then
    MODEL_DIGEST=$(sha256sum "$MODEL_DIR/config.json" | cut -d' ' -f1)
    echo "[confidential-ai] Model digest (config.json): $MODEL_DIGEST"
  fi
fi

# Derive a display name from the model path for metadata
DISPLAY_NAME="$MODEL"
if [[ "$DISPLAY_NAME" == /models/* ]]; then
  DISPLAY_NAME="${DISPLAY_NAME#/models/}"
fi

# Register model digest with the enclave OS manager so RA-TLS
# certificates include OID 3.5. Uses the internal API (localhost:9444)
# which is unauthenticated within the VM trust boundary.
CONTAINER_NAME="${ENCLAVE_OS_CONTAINER_NAME:-}"
if [[ -n "$MODEL_DIGEST" && -n "$CONTAINER_NAME" ]]; then
  echo "[confidential-ai] Registering model digest with enclave OS (OID 3.5)"
  curl -sf -X PUT \
    "http://127.0.0.1:9444/api/v1/containers/${CONTAINER_NAME}/extensions" \
    -H "Content-Type: application/json" \
    -d "{\"model_digest\":\"${MODEL_DIGEST}\"}" \
    || echo "[confidential-ai] WARNING: failed to register model digest (manager may not be ready yet)"
fi

# Start the Go proxy server
echo "[confidential-ai] Starting reproducibility proxy on $PROXY_PORT"
exec /usr/local/bin/confidential-ai \
  --listen "$PROXY_PORT" \
  --vllm-upstream "http://localhost:$VLLM_PORT" \
  --model "$DISPLAY_NAME" \
  --quantization "$QUANT" \
  --model-digest "$MODEL_DIGEST"
