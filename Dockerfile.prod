FROM golang:1.22-bookworm AS builder

WORKDIR /src
COPY go.mod go.sum ./
RUN go mod download
COPY . .
RUN CGO_ENABLED=0 go build -trimpath -ldflags='-s -w' -o /confidential-ai ./cmd/server/

# ---

FROM nvidia/cuda:12.6.3-runtime-ubuntu24.04

ARG VLLM_VERSION=0.19.0

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir --break-system-packages \
    vllm==${VLLM_VERSION}

COPY --from=builder /confidential-ai /usr/local/bin/confidential-ai
COPY scripts/entrypoint.sh /opt/confidential-ai/entrypoint.sh
RUN chmod +x /opt/confidential-ai/entrypoint.sh

EXPOSE 8000 8080

ENTRYPOINT ["/opt/confidential-ai/entrypoint.sh"]
