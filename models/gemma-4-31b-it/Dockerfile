# Gemma 4 31B IT - Confidential Inference
#
# Dense 30.7B parameter model from Google. Fits on a single H100 80GB in BF16.
# Model weights are pre-loaded on a GCP persistent disk (model-gemma4-31b).
# The manager reads the ai.privasys.volume label and bind-mounts the disk
# into the container at /models.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest .

FROM ghcr.io/privasys/confidential-ai:latest

LABEL ai.privasys.volume="/mnt/model-gemma4-31b:/models:ro"

ENV MODEL_URL=file:///models
ENV DTYPE=bfloat16

EXPOSE 8000 8080
