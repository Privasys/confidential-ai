# Gemma 4 31B IT - Confidential Inference
#
# Dense 30.7B parameter model from Google. Fits on a single H100 80GB in BF16.
# Model is downloaded from HuggingFace at first startup (~62 GB).
#
# Requires vLLM main branch: the gemma4 architecture is not in any release yet.
# Once a vLLM release includes Gemma 4 support, switch back to a pinned version.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest .
#
# Run (standalone, requires NVIDIA GPU):
#   docker run --gpus all -p 8000:8000 \
#     -e HF_TOKEN=<token> \
#     -v model-cache:/root/.cache/huggingface \
#     ghcr.io/privasys/confidential-ai-gemma4:latest

FROM ghcr.io/privasys/confidential-ai:latest

ENV MODEL_NAME=google/gemma-4-31b-it
ENV DTYPE=bfloat16

EXPOSE 8000 8080
