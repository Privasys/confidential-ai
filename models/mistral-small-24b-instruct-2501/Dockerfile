# Mistral Small 24B Instruct - Confidential Inference
#
# Dense 24B parameter model from Mistral AI. Fits on a single H100 80GB in
# BF16. Model weights are pre-loaded on a GCP persistent disk and bind-mounted
# into the container at /models/.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-mistral-small:latest .

FROM ghcr.io/privasys/confidential-ai:latest

ENV MODEL_URL=file:///models/mistralai/Mistral-Small-24B-Instruct-2501
ENV DTYPE=bfloat16

EXPOSE 8000 8080
