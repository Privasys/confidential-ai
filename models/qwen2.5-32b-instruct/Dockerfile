# Qwen 2.5 32B Instruct - Confidential Inference
#
# Dense 32B parameter model from Alibaba / Qwen team. Fits on a single H100
# 80GB in BF16. Model weights are pre-loaded on a GCP persistent disk
# (model-qwen25-32b). The manager reads the ai.privasys.volume label and
# bind-mounts the disk into the container at /models.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-qwen25:latest .

FROM ghcr.io/privasys/confidential-ai:latest

LABEL ai.privasys.volume="/mnt/model-qwen25-32b:/models:ro"

ENV MODEL_URL=file:///models
ENV DTYPE=bfloat16

EXPOSE 8000 8080
