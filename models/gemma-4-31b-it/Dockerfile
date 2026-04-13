# Gemma 4 31B IT - Confidential Inference
#
# Dense 30.7B parameter model from Google. Fits on a single H100 80GB in BF16.
# Model weights are pre-loaded on a GCP persistent disk and bind-mounted into
# the container at /models/ by the management service (see ai-gpu-startup.sh).
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-gemma4:latest .

FROM ghcr.io/privasys/confidential-ai:latest

ENV MODEL_URL=file:///models/google/gemma-4-31b-it
ENV DTYPE=bfloat16

EXPOSE 8000 8080
