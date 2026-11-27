#!/usr/bin/env bash
# Runs the batch-invariance suite once, step by step, each step's output in
# /data/logs/<step>.log and its result in /data/logs/status.json, then keeps
# serving the logs. STEPS (space-separated) picks the steps to run.
set -u
# Model weights (public) and compile caches live on the container filesystem
# (the guest workspace disk): Imperial's volume pool has ~20 GB free. Only
# the logs go on the encrypted /data volume.
export HOME=/opt/home HF_HOME=/opt/home/.cache/huggingface
mkdir -p "$HOME"
# tests/ imports tests.utils; the source copy of vllm/ was removed in the
# image, so `import vllm` still resolves to the patched installed wheel.
export PYTHONPATH=/opt/vllm-src
LOGS=/data/logs
mkdir -p "$LOGS"
python3 /opt/bi-test/serve.py "$LOGS" "${PORT:-8080}" &

# This rerun: the rebased branch, the norm test, and the suite with and
# without FlashInfer kernels. The full list: env sched-unit norm-unit
# norm-unit-nofix main-0.8b bi-0.8b bi-0.8b-nofi bi-36 bi-38 lm-0 lm-1
# bench-0 bench-1.
STEPS=${STEPS:-"env sched-unit norm-unit norm-unit-nofix bi-0.8b bi-0.8b-nofi bi-36"}
M36=Qwen/Qwen3.6-35B-A3B-FP8
M38=Qwen/Qwen3.8-27B-FP8
cd /opt/vllm-src/tests

status() { # step state [exit]
  python3 - "$LOGS/status.json" "$1" "$2" "${3:-}" <<'PY'
import json, sys, time, os
path, step, state, code = sys.argv[1:5]
d = json.load(open(path)) if os.path.exists(path) else {}
d[step] = {"state": state, "exit": code, "at": time.strftime("%H:%M:%SZ", time.gmtime())}
json.dump(d, open(path, "w"), indent=1)
PY
}

RERUN=$(cat /opt/bi-test/rerun 2>/dev/null)

run() { # step command...
  local step=$1; shift
  case " $STEPS " in *" $step "*) ;; *) return ;; esac
  # A redeploy resumes: a step already done is skipped unless the image
  # asks for it again (/opt/bi-test/rerun).
  case " $RERUN " in *" $step "*) ;; *)
    if python3 -c "import json,sys; sys.exit(0 if json.load(open('$LOGS/status.json')).get('$step',{}).get('state')=='done' else 1)" 2>/dev/null; then
      return
    fi ;;
  esac
  status "$step" running
  echo "\$ $*" > "$LOGS/$step.log"
  "$@" >> "$LOGS/$step.log" 2>&1
  local code=$?
  status "$step" done "$code"
}

lm() { # bi: gsm8k as the chat model is served, thinking off
  VLLM_BATCH_INVARIANT=$1 lm_eval --model vllm \
    --model_args pretrained=$M36,max_model_len=8192,gpu_memory_utilization=0.85,enable_thinking=False \
    --tasks gsm8k --num_fewshot 5 --apply_chat_template --fewshot_as_multiturn --batch_size auto
}

bench() { # bi
  VLLM_BATCH_INVARIANT=$1 vllm serve $M36 --max-model-len 8192 --port 8100 > "$LOGS/bench-$1.server.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 240); do curl -sf localhost:8100/health >/dev/null && break; sleep 5; done
  vllm bench serve --model $M36 --port 8100 --dataset-name random --random-input-len 1024 \
    --random-output-len 256 --num-prompts 500 --max-concurrency 32 --seed 0
  local code=$?
  kill $pid; wait $pid 2>/dev/null
  return $code
}

SITE=/usr/local/lib/python3.12/dist-packages
# The scheduler side of GDN chunked-prefill invariance, with the Mamba split
# tests in the same file.
SCHED_UNIT="v1/core/test_mamba_align_chunk_split.py"

main_bi() { # model: the same suite on main (the diff reversed, main's tests/)
  patch -R -p1 -d $SITE --no-backup-if-mismatch < /opt/gdn-bi.patch || return 1
  ( cd /opt/vllm-main/tests && PYTHONPATH=/opt/vllm-main VLLM_TEST_MODEL=$1       pytest -v -s v1/determinism/test_batch_invariance.py )
  local code=$?
  patch -p1 -d $SITE --no-backup-if-mismatch < /opt/gdn-bi.patch
  return $code
}

NORM_UNIT="models/qwen3_5/test_gemma_add_rms_norm_quant.py -k batch_invariant"

norm_nofix() { # the norm test against the branch without its warp pin: must fail
  local f=$SITE/vllm/models/qwen3_5/nvidia/ops/fused_norm_quant.py
  cp $f /tmp/fused_norm_quant.py
  sed -i 's/ and not envs.VLLM_BATCH_INVARIANT//' $f
  grep -nE "num_warps=16|use_16_warps =" $f
  pytest -v $NORM_UNIT
  local code=$?
  cp /tmp/fused_norm_quant.py $f
  return $code
}

nofi() { # model: the suite without FlashInfer kernels (no flashinfer-cubin, no nvcc), as on a precompiled install
  local d
  d=$(python3 -c "import importlib.util as u, os; print(os.path.dirname(u.find_spec('flashinfer_cubin').origin))")
  mv "$d" "$d.off"
  CUDA_HOME=/nonexistent python3 -c "from vllm.utils.flashinfer import has_flashinfer; print('has_flashinfer', has_flashinfer())"
  CUDA_HOME=/nonexistent VLLM_TEST_MODEL=$1 pytest -v -s v1/determinism/test_batch_invariance.py
  local code=$?
  mv "$d.off" "$d"
  return $code
}

run env bash -c 'nvidia-smi; python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__)"; cat /opt/gdn-bi.commits'
run sched-unit pytest -v $SCHED_UNIT
run norm-unit pytest -v $NORM_UNIT
run norm-unit-nofix norm_nofix
# The suite's sliding-window test uses gated google/gemma-2-2b-it: wait for
# the token (POST /hf-token, kept on /data across redeploys).
hf_token() {
  local f=/data/secrets/hf-token
  for _ in $(seq 1 360); do [ -s $f ] && break; sleep 5; done
  [ -s $f ] && export HF_TOKEN=$(cat $f)
}
case " $STEPS " in *" bi-"*) hf_token ;; esac
run main-0.8b main_bi Qwen/Qwen3.5-0.8B
run bi-0.8b env VLLM_TEST_MODEL=Qwen/Qwen3.5-0.8B pytest -v -s v1/determinism/test_batch_invariance.py
run bi-0.8b-nofi nofi Qwen/Qwen3.5-0.8B
run bi-36 env VLLM_TEST_MODEL=$M36 VLLM_GPU_MEMORY_UTILIZATION=0.85 pytest -v -s v1/determinism/test_batch_invariance.py
run bi-38 env VLLM_TEST_MODEL=$M38 VLLM_GPU_MEMORY_UTILIZATION=0.85 pytest -v -s v1/determinism/test_batch_invariance.py
run lm-0 lm 0
run lm-1 lm 1
run bench-0 bench 0
run bench-1 bench 1
status all done
wait
