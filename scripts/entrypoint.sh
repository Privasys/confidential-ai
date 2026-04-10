#!/bin/bash
set -euo pipefail

MODEL="${MODEL_NAME:-}"
if [[ -z "$MODEL" ]]; then
  echo "[confidential-ai] ERROR: MODEL_NAME environment variable is required"
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

# Start the Go proxy server
echo "[confidential-ai] Starting reproducibility proxy on $PROXY_PORT"
exec /usr/local/bin/confidential-ai \
  --listen "$PROXY_PORT" \
  --vllm-upstream "http://localhost:$VLLM_PORT" \
  --model "$MODEL" \
  --quantization "$QUANT"
