# Llama 4 Scout 17Bx16E Instruct - Confidential Inference
#
# MoE model from Meta with 109B total params (17B active, 16 experts).
# Requires on-the-fly INT4 quantization to fit on a single H100 80GB (~58 GB VRAM).
# Model weights (BF16) are pre-loaded on a GCP persistent disk (model-llama4-scout).
# The manager reads the ai.privasys.volume label and bind-mounts the disk
# into the container at /models.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-llama4-scout:latest .

FROM ghcr.io/privasys/confidential-ai:latest

LABEL ai.privasys.volume="/mnt/model-llama4-scout:/models:ro"

ENV MODEL_URL=file:///models
ENV DTYPE=bfloat16
ENV QUANTIZATION=int4

EXPOSE 8000 8080
