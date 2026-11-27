# Night-test harness for the GDN batch-invariance patch (Privasys/vllm branch
# gdn-batch-invariance), run on Imperial's H100 while Confidential AI is
# stopped. Not a product image: it runs the upstream reviewer's suite once
# and serves the logs.
#
# vLLM is the upstream per-commit wheel at the branch's base commit, with the
# branch's Python diff applied on top (the patch touches Python only), the
# same way the production image patches its release wheel.
FROM nvidia/cuda:13.0.3-devel-ubuntu24.04

ARG VLLM_BASE=9172ffc42e16e3a276ab8519c6d6a949c0f1d76c
ARG VLLM_WHEEL=vllm-0.34.1rc1.dev292%2Bg9172ffc42-cp38-abi3-manylinux_2_28_x86_64.whl
# The rebased PR branch, pushed under its own name until its tests pass.
ARG BRANCH=gdn-bi-rerun

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3.12 python3.12-venv python3.12-dev python3-pip ca-certificates curl git patch gcc libc6-dev && \
    rm -rf /var/lib/apt/lists/*
ENV UV_BREAK_SYSTEM_PACKAGES=1

# FlashInfer pinned by URL as in the production image; only the H100
# (sm90a) jit-cache provider, with the wrapper, without dependencies.
# main pins torchvision 0.29.2 and torchaudio 2.11.1, which have no Linux
# wheel published anywhere yet; the cu130 index's 0.29.1 and 2.11.0 are the
# matching releases for torch 2.14.
RUN pip3 install --no-cache-dir --break-system-packages uv && \
    printf 'torchvision==0.29.1\ntorchaudio==2.11.0\n' > /tmp/overrides.txt && \
    uv pip install --system --torch-backend=cu130 --override /tmp/overrides.txt \
      "https://wheels.vllm.ai/${VLLM_BASE}/${VLLM_WHEEL}" \
      "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.4/flashinfer_cubin-0.7.4-py3-none-any.whl" && \
    uv pip install --system --no-deps \
      "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.4/flashinfer_jit_cache-0.7.4+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" \
      "https://github.com/flashinfer-ai/flashinfer/releases/download/v0.7.4/flashinfer_jit_cache_sm90a-0.7.4+cu130-cp39-abi3-manylinux_2_28_x86_64.whl" && \
    uv pip install --system pytest pytest-asyncio pytest-timeout anthropic tblib cloudpickle httpx requests openai "lm-eval==0.4.13"

# The branch's tests/ tree and its diff applied onto the wheel; main's own
# tests/ tree too, for the "main" run (with the diff reversed).
# The branch head (set by the workflow), so a moved branch invalidates the
# clone below (BuildKit otherwise reuses the cached layer).
ARG BI_HEAD=unset
ARG KV_HEAD=unset
RUN echo "head: $BI_HEAD" && git clone --depth 4 --branch ${BRANCH} https://github.com/Privasys/vllm.git /opt/vllm-src &&     cd /opt/vllm-src && git fetch --depth 1 origin ${VLLM_BASE} &&     git diff ${VLLM_BASE} HEAD -- vllm/ > /opt/gdn-bi.patch &&     git log --oneline ${VLLM_BASE}..HEAD > /opt/gdn-bi.commits &&     mkdir -p /opt/vllm-main && git archive ${VLLM_BASE} tests | tar -x -C /opt/vllm-main &&     cd /usr/local/lib/python3.12/dist-packages &&     patch -p1 --forward --no-backup-if-mismatch < /opt/gdn-bi.patch &&     grep -q 'class QwenGDNAttentionBackend' vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py &&     rm -rf /opt/vllm-src/vllm

# As in the production image: the Qwen Triton warmup's synthetic GDN shapes
# hang a kernel launch on the H100 in CC mode, and DeepGEMM's FP8 JIT is off.
RUN python3 - <<'PYEOF'
import pathlib
p = pathlib.Path("/usr/local/lib/python3.12/dist-packages/vllm/model_executor/warmup/qwen_triton_warmup.py")
src = p.read_text()
assert "def qwen_triton_warmup(" in src
p.write_text(src + "\n\ndef qwen_triton_warmup(runner, model_config):  # noqa: F811\n    return None\n")
PYEOF
ENV VLLM_USE_DEEP_GEMM=0 VLLM_MOE_USE_DEEP_GEMM=0

COPY run.sh serve.py rerun /opt/bi-test/
RUN chmod +x /opt/bi-test/run.sh

LABEL org.opencontainers.image.source="https://github.com/Privasys/vllm"
LABEL ai.privasys.devices="/dev/nvidia0,/dev/nvidiactl,/dev/nvidia-uvm,/dev/nvidia-uvm-tools"
ENTRYPOINT ["/opt/bi-test/run.sh"]
