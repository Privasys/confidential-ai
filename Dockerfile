# Confidential AI - Reproducible Inference Proxy
#
# This Dockerfile builds the Go proxy that wraps vLLM and injects reproducibility
# metadata into every inference response. The proxy is designed to run inside a
# TDX Confidential VM alongside a vLLM backend.
#
# For production with GPU + vLLM, see Dockerfile.prod.

FROM golang:1.22-bookworm AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /confidential-ai ./cmd/server/

# ---

FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

COPY --from=builder /confidential-ai /usr/local/bin/confidential-ai

EXPOSE 8080

CMD ["confidential-ai", \
     "--listen", ":8080", \
     "--vllm-upstream", "http://localhost:8000", \
     "--tee-type", "tdx"]
