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

STEPS=${STEPS:-"env kv-unit kv-before kv bi-0.8b bi-36 bi-38 lm-0 lm-1 bench-0 bench-1"}
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

kv() { # the CPU KV tier under CC: starts, byte-identical hits, timings
  # Dev mode exposes /reset_prefix_cache: the test clears the GPU prefix
  # cache so its repeat request can only be served from the CPU tier.
  VLLM_SERVER_DEV_MODE=1 VLLM_BATCH_INVARIANT=1 VLLM_USE_SIMPLE_KV_OFFLOAD=1 vllm serve $M36 --port 8100 \
    --max-model-len 16384 --max-num-seqs 4 --enable-prefix-caching --kv-cache-dtype fp8 \
    --kv-offloading-size 16 \
    --default-chat-template-kwargs '{"enable_thinking": false}' > "$LOGS/kv.server.log" 2>&1 &
  local pid=$!
  for _ in $(seq 1 360); do curl -sf localhost:8100/health >/dev/null && break; kill -0 $pid 2>/dev/null || break; sleep 5; done
  python3 /opt/bi-test/kv_test.py http://localhost:8100 $M36
  local code=$?
  kill $pid; wait $pid 2>/dev/null
  return $code
}

SITE=/usr/local/lib/python3.12/dist-packages
unpatch_kv() { patch -R -p1 -d $SITE --no-backup-if-mismatch < /opt/kv-cc.patch; }
repatch_kv() { patch -p1 -d $SITE --no-backup-if-mismatch < /opt/kv-cc.patch; }
KV_UNIT=v1/kv_connector/unit/test_simple_cpu_offload_cc.py

kv_unit() { # the unit test passes with the CC patch and fails without it
  pytest -v $KV_UNIT; local with=$?
  unpatch_kv && { pytest -v $KV_UNIT; local without=$?; repatch_kv; }
  echo "== with the patch: exit $with; without the patch: exit ${without:-?}"
  [ "$with" = 0 ] && [ "${without:-0}" != 0 ]
}

kv_before() { # on this hardware, the unpatched engine fails to start the CPU tier
  unpatch_kv || return 1
  VLLM_USE_SIMPLE_KV_OFFLOAD=1 timeout 1800 vllm serve $M36 --port 8100 \
    --max-model-len 16384 --max-num-seqs 4 --enable-prefix-caching --kv-cache-dtype fp8 \
    --num-gpu-blocks-override 64 --kv-offloading-size 16
  local code=$?
  repatch_kv
  echo "== unpatched engine exit $code (expect non-zero with cudaHostRegister failed above)"
  [ "$code" != 0 ]
}

run env bash -c 'nvidia-smi; python3 -c "import vllm, torch; print(vllm.__version__, torch.__version__)"; cat /opt/gdn-bi.commits'
run kv-unit kv_unit
run kv-before kv_before
run kv kv
run bi-0.8b env VLLM_TEST_MODEL=Qwen/Qwen3.5-0.8B pytest -v -s v1/determinism/test_batch_invariance.py
run bi-36 env VLLM_TEST_MODEL=$M36 VLLM_GPU_MEMORY_UTILIZATION=0.85 pytest -v -s v1/determinism/test_batch_invariance.py
run bi-38 env VLLM_TEST_MODEL=$M38 VLLM_GPU_MEMORY_UTILIZATION=0.85 pytest -v -s v1/determinism/test_batch_invariance.py
run lm-0 env VLLM_BATCH_INVARIANT=0 lm_eval --model vllm --model_args pretrained=$M36,max_model_len=8192,gpu_memory_utilization=0.85 --tasks gsm8k --num_fewshot 5 --batch_size auto
run lm-1 env VLLM_BATCH_INVARIANT=1 lm_eval --model vllm --model_args pretrained=$M36,max_model_len=8192,gpu_memory_utilization=0.85 --tasks gsm8k --num_fewshot 5 --batch_size auto
run bench-0 bench 0
run bench-1 bench 1
status all done
wait
