# Qwen 2.5 32B Instruct - Confidential Inference
#
# Dense 32B parameter model from Alibaba / Qwen team. Fits on a single H100
# 80GB in BF16. Model weights are pre-loaded on a GCP persistent disk and
# bind-mounted into the container at /models/.
#
# Build:
#   docker build -t ghcr.io/privasys/confidential-ai-qwen25:latest .

FROM ghcr.io/privasys/confidential-ai:latest

ENV MODEL_URL=file:///models/Qwen/Qwen2.5-32B-Instruct
ENV DTYPE=bfloat16

EXPOSE 8000 8080
