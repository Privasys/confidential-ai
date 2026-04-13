# Mistral Small 24B Instruct - Confidential Inference
#
# Dense 24B parameter model from Mistral AI. Fits on a single H100 80GB in
# BF16. Model weights are pre-loaded on a GCP persistent disk
# (model-mistral-small-24b). The manager reads the ai.privasys.volume label
# and bind-mounts the disk into the container at /models.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-mistral-small:latest .

FROM ghcr.io/privasys/confidential-ai:latest

LABEL ai.privasys.volume="/mnt/model-mistral-small-24b:/models:ro"

ENV MODEL_URL=file:///models
ENV DTYPE=bfloat16

EXPOSE 8000 8080
