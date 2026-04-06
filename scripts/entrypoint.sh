#!/bin/bash
set -euo pipefail

MODEL="${MODEL_NAME:-gpt-oss-120b}"
QUANT="${QUANTIZATION:-awq}"
VLLM_PORT="${VLLM_PORT:-8000}"
PROXY_PORT="${LISTEN_ADDR:-:8080}"

echo "[confidential-ai] Starting vLLM for model=$MODEL quantization=$QUANT"

# Start vLLM in the background
vllm serve "$MODEL" \
  --seed 0 \
  --tensor-parallel-size 1 \
  --quantization "$QUANT" \
  --enable-batch-invariance \
  --disable-log-requests \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.92 \
  --port "$VLLM_PORT" &

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
