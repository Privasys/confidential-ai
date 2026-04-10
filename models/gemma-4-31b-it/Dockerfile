# Gemma 4 31B IT - Confidential Inference
#
# Dense 30.7B parameter model from Google. Fits on a single H100 80GB in BF16.
# Model is downloaded from HuggingFace at first startup (~62 GB).
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest .
#
# Run (standalone, requires NVIDIA GPU):
#   docker run --gpus all -p 8000:8000 \
#     -e HF_TOKEN=<token> \
#     -v model-cache:/root/.cache/huggingface \
#     ghcr.io/privasys/confidential-ai-gemma4:latest

FROM vllm/vllm-openai:v0.19.0

# Reproducibility: force V0 engine and deterministic CUDA
ENV VLLM_USE_V1=0
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8
ENV PYTHONHASHSEED=0

EXPOSE 8000

CMD ["--model", "google/gemma-4-31b-it", \
     "--seed", "0", \
     "--tensor-parallel-size", "1", \
     "--enforce-eager", \
     "--no-enable-log-requests", \
     "--max-model-len", "8192", \
     "--gpu-memory-utilization", "0.90", \
     "--dtype", "bfloat16", \
     "--port", "8000"]
