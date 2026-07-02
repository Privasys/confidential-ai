# Confidential AI - Reproducible Inference Proxy
#
# This Dockerfile builds the Go proxy that wraps vLLM and injects reproducibility
# metadata into every inference response. The proxy is designed to run inside a
# TDX Confidential VM alongside a vLLM backend.
#
# For production with GPU + vLLM, see Dockerfile.prod.

FROM golang:1.22-bookworm AS builder

# The MCP RA-TLS transport needs (a) the Privasys Go fork for the
# ClientHello challenge extension (-tags ratls) and (b) the sibling
# ra-tls-clients module that go.mod replaces to ../ra-tls-clients/go.
# Keep both pins in sync with .github/workflows/build.yml.
ARG GO_RATLS_VERSION=privasys-v0.2.0-go1.25.8
ARG RA_TLS_CLIENTS_REF=8a0318d2641ff4e4ce7e8cbaa8391b04fdbb48c9
RUN curl -sL "https://github.com/Privasys/go/releases/download/${GO_RATLS_VERSION}/go-ratls-${GO_RATLS_VERSION}-linux-amd64.tar.gz"       -o /tmp/go-ratls.tar.gz  && tar -C /usr/local -xzf /tmp/go-ratls.tar.gz  && rm /tmp/go-ratls.tar.gz  && git clone https://github.com/Privasys/ra-tls-clients /ra-tls-clients  && git -C /ra-tls-clients checkout "${RA_TLS_CLIENTS_REF}"
ENV GOROOT=/usr/local/go-ratls
ENV PATH=/usr/local/go-ratls/bin:${PATH}
# The fork's release tarball ships a hermetic go.env (GOPROXY with no
# usable entries); restore standard module resolution explicitly.
ENV GOPROXY=https://proxy.golang.org,direct GOSUMDB=sum.golang.org GOTOOLCHAIN=local

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -tags ratls -trimpath -ldflags='-s -w' -o /confidential-ai ./cmd/server/

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
