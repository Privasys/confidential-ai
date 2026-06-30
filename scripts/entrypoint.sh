#!/bin/bash
set -euo pipefail

# --- Confidential-AI Entrypoint ------------------------------------------
#
# Two modes:
#   1. Dynamic model loading (new): Go proxy starts immediately, vLLM is
#      started on-demand via POST /v1/models/load. Requires MODELS_DIR.
#   2. Legacy mode: vLLM starts at boot from MODEL_URL/MODEL_NAME, proxy
#      polls /health until ready. Used by per-model Dockerfiles.
#
# The mode is auto-detected: if MODELS_DIR is set and points to a directory
# with model subdirectories, dynamic mode is used. Otherwise legacy mode.

MODELS_DIR="${MODELS_DIR:-}"
MODEL_URL="${MODEL_URL:-}"
MODEL_NAME="${MODEL_NAME:-}"
# Listen on the platform-allocated port. The host injects PORT; honour an
# explicit LISTEN_ADDR override first, otherwise :$PORT. There is no hard-coded
# fallback port — PORT (or LISTEN_ADDR) is required (the manager health check
# probes the allocated port; a fixed guess would break it).
PROXY_PORT="${LISTEN_ADDR:-${PORT:+:$PORT}}"
if [[ -z "$PROXY_PORT" ]]; then
  echo "[confidential-ai] ERROR: PORT (or LISTEN_ADDR) is required" >&2
  exit 1
fi

# Reproducibility environment (applies to both modes).
#
# CUBLAS_WORKSPACE_CONFIG is required by PyTorch's deterministic
# algorithms path under CUDA >= 10.2; PYTHONHASHSEED removes the
# only source of Python-level non-determinism we care about.
#
# VLLM_USE_V1 is intentionally NOT set here. We used to force V0
# for batch-invariance, but V0 leaves the H100 tensor cores idle.
# Determinism on V1 is preserved by disabling chunked prefill
# (see --no-enable-chunked-prefill below) and sizing
# --max-num-batched-tokens so any prompt fits in a single prefill.
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONHASHSEED=0

# --- Dynamic mode: just start the Go proxy --------------------------------
if [[ -n "$MODELS_DIR" && -d "$MODELS_DIR" ]]; then
  echo "[confidential-ai] Dynamic model loading mode (models_dir=$MODELS_DIR)"
  exec /usr/local/bin/confidential-ai \
    --listen "$PROXY_PORT" \
    --models-dir "$MODELS_DIR" \
    --roothash-dir "${ROOTHASH_DIR:-/var/lib/enclave-os/model-roothashes}" \
    --load-token "${LOAD_TOKEN:-}" \
    --gpu-type "${GPU_TYPE:-H100-80GB}" \
    --tee-type "${TEE_TYPE:-tdx}" \
    --cuda-version "${CUDA_VERSION:-12.6.3}" \
    --vllm-version "${VLLM_VERSION:-0.22.1}" \
    --image-digest "${IMAGE_DIGEST:-}" \
    --mcp-servers "${MCP_SERVERS:-}" \
    --tool-spec-url "${TOOL_SPEC_URL:-}" \
    --tool-spec-token "${TOOL_SPEC_TOKEN:-}" \
    --tool-spec-interval "${TOOL_SPEC_INTERVAL:-60s}"
fi

# --- Legacy mode: start vLLM at boot with MODEL_URL/MODEL_NAME -----------
if [[ -n "$MODEL_URL" ]]; then
  case "$MODEL_URL" in
    file://*)
      MODEL="${MODEL_URL#file://}"
      export HF_HUB_OFFLINE=1
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
  echo "[confidential-ai] ERROR: MODELS_DIR, MODEL_URL, or MODEL_NAME is required"
  exit 1
fi
QUANT="${QUANTIZATION:-}"
DTYPE="${DTYPE:-auto}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
GPU_MEM="${GPU_MEMORY_UTILIZATION:-0.90}"
VLLM_PORT="${VLLM_PORT:-8000}"

echo "[confidential-ai] Legacy mode: starting vLLM for model=$MODEL quantization=$QUANT dtype=$DTYPE"

# Build vLLM args. CUDA graphs ENABLED (no --enforce-eager) and
# V1 scheduler ENABLED (default), with chunked prefill OFF and the
# batched-token budget sized so any prompt up to --max-model-len
# is processed in one mathematical block. See manager.go for
# the full reproducibility rationale.
BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-}"
if [[ -z "$BATCHED_TOKENS" ]]; then
  BATCHED_TOKENS=$((MAX_MODEL_LEN * 2))
  if (( BATCHED_TOKENS < 16384 )); then
    BATCHED_TOKENS=16384
  fi
fi

VLLM_ARGS=(
  --seed 0
  --tensor-parallel-size 1
  --no-enable-chunked-prefill
  --max-num-batched-tokens "$BATCHED_TOKENS"
  --no-enable-log-requests
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM"
  --dtype "$DTYPE"
  --port "$VLLM_PORT"
)

# SERVED_MODEL_NAME pins the canonical id vLLM exposes (so chat
# clients and reproducibility metadata see a friendly short name
# instead of a filesystem path). Defaults to MODEL_NAME for
# backwards compatibility, then falls back to whatever vLLM picks
# from the model path.
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-${MODEL_NAME:-}}"
if [[ -n "$SERVED_MODEL_NAME" ]]; then
  VLLM_ARGS+=(--served-model-name "$SERVED_MODEL_NAME")
fi

if [[ -n "$QUANT" && "$QUANT" != "none" ]]; then
  VLLM_ARGS+=(--quantization "$QUANT")
fi

# Start vLLM in the background
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

echo "[confidential-ai] Starting reproducibility proxy on $PROXY_PORT"
exec /usr/local/bin/confidential-ai \
  --listen "$PROXY_PORT" \
  --vllm-upstream "http://localhost:$VLLM_PORT" \
  --model "$DISPLAY_NAME" \
  --quantization "$QUANT" \
  --model-digest "$MODEL_DIGEST"
