# Confidential AI - Reproducible Inference Proxy
#
# This Dockerfile builds the Go proxy that wraps vLLM and injects reproducibility
# metadata into every inference response. The proxy is designed to run inside a
# TDX Confidential VM alongside a vLLM backend.
#
# For production with GPU + vLLM, see Dockerfile.prod.

FROM golang:1.26-bookworm AS builder

# The MCP RA-TLS v2 transport builds on upstream Go; it needs the sibling
# ra-tls-clients module that go.mod replaces to ../ra-tls-clients/go.
# Keep the pin in sync with .github/workflows/build.yml (an RA-TLS v2 commit).
ARG RA_TLS_CLIENTS_REF=64438d2593f26e4250bbb75e25240a5422876136
RUN git clone https://github.com/Privasys/ra-tls-clients /ra-tls-clients  && git -C /ra-tls-clients checkout "${RA_TLS_CLIENTS_REF}"
ENV GOTOOLCHAIN=local

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

# No EXPOSE and no --listen: the platform injects $PORT and the proxy binds it
# ($PORT is required, no hard-coded fallback). Host networking ignores EXPOSE.
CMD ["confidential-ai", \
     "--vllm-upstream", "http://localhost:8000", \
     "--tee-type", "tdx"]
