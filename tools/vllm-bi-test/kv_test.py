"""CPU KV tier check against a local `vllm serve` (see run.sh, step kv).

The server runs with VLLM_SERVER_DEV_MODE=1. After a cold request, the GPU
prefix cache is cleared (POST /reset_prefix_cache, which keeps the CPU tier),
so the same request can only hit the CPU tier. Checks:
  1. the seeded request is byte-identical served cold, from the CPU tier, and
     again from the GPU;
  2. the offload connector loaded blocks back from the CPU tier;
  3. time to first token, cold against CPU hit against GPU hit.
Exit status 0 only if 1 and 2 hold.
"""

import json
import random
import re
import sys
import time
import urllib.request

BASE = sys.argv[1]  # e.g. http://localhost:8100
MODEL = sys.argv[2]


def prompt(seed: int, n_words: int = 6000) -> str:
    rnd = random.Random(seed)
    words = ["enclave", "attestation", "kernel", "vault", "quote", "measure",
             "tensor", "cache", "prefix", "token", "secure", "replay"]
    return " ".join(rnd.choice(words) for _ in range(n_words)) + "\nSummarise the text above in two sentences."


def chat(text: str) -> tuple[str, float]:
    body = json.dumps({
        "model": MODEL, "seed": 1234, "temperature": 0.7, "max_tokens": 64, "stream": True,
        "messages": [{"role": "user", "content": text}],
        "chat_template_kwargs": {"enable_thinking": False},
    }).encode()
    req = urllib.request.Request(BASE + "/v1/chat/completions", body, {"Content-Type": "application/json"})
    t0 = time.time()
    ttft, out = None, []
    with urllib.request.urlopen(req, timeout=600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            delta = json.loads(line[6:])["choices"][0]["delta"].get("content") or ""
            if delta and ttft is None:
                ttft = time.time() - t0
            out.append(delta)
    return "".join(out), ttft or 0.0


def offload_metrics() -> dict[str, float]:
    """Every simple_kv_offload / external prefix-cache / cached-by-source sample."""
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    out: dict[str, float] = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        if re.match(r"vllm:(simple_kv_offload|external_prefix_cache|prompt_tokens_cached_by_source)", line):
            name, _, value = line.rpartition(" ")
            try:
                out[name] = float(value)
            except ValueError:
                pass
    return out


def reset_gpu_prefix_cache() -> None:
    req = urllib.request.Request(BASE + "/reset_prefix_cache", b"", method="POST")
    urllib.request.urlopen(req, timeout=60).read()


target = prompt(1)
cold, t_cold = chat(target)
time.sleep(2)  # let the store to the CPU tier finish
before = offload_metrics()
reset_gpu_prefix_cache()
cpu_hit, t_cpu = chat(target)
after = offload_metrics()
gpu_hit, t_gpu = chat(target)

changed = {k: after.get(k, 0) - before.get(k, 0) for k in after if after.get(k, 0) != before.get(k, 0)}
loaded = any(("load" in k or "hit" in k or 'source="host"' in k or "cpu" in k) and v > 0 for k, v in changed.items())
same = cold == cpu_hit == gpu_hit

print(f"cold      ttft {t_cold:.3f}s")
print(f"cpu hit   ttft {t_cpu:.3f}s")
print(f"gpu hit   ttft {t_gpu:.3f}s")
print("offload counters that moved on the CPU-hit request:")
for k, v in sorted(changed.items()):
    print(f"  {k} +{v:g}")
print(f"byte-identical: {same}   cpu tier read back: {loaded}")
if not same:
    print("cold   :", repr(cold))
    print("cpu hit:", repr(cpu_hit))
    print("gpu hit:", repr(gpu_hit))
sys.exit(0 if same and loaded else 1)
