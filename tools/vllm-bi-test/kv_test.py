"""CPU KV tier check against a local `vllm serve` (see run.sh, step kv).

The server runs with a deliberately small GPU KV cache, so a long prompt's
blocks are evicted to the CPU tier by a few other long prompts. Then:
  1. the same seeded request is byte-identical served cold, from the CPU tier,
     and again (GPU hit);
  2. the CPU-tier hit counter rises;
  3. time to first token, cold against CPU hit.
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


def metric(name: str) -> float:
    text = urllib.request.urlopen(BASE + "/metrics", timeout=30).read().decode()
    total = 0.0
    for line in text.splitlines():
        if line.startswith(name):
            m = re.search(r"\s([0-9.eE+-]+)$", line)
            if m:
                total += float(m.group(1))
    return total


target = prompt(1)
cold, t_cold = chat(target)
for s in range(2, 8):  # evict the target's blocks to the CPU tier
    chat(prompt(s))
host_before = metric('vllm:prompt_tokens_cached_by_source_total{source="host"') + metric("vllm:external_prefix_cache_hits_total")
cpu_hit, t_cpu = chat(target)
host_after = metric('vllm:prompt_tokens_cached_by_source_total{source="host"') + metric("vllm:external_prefix_cache_hits_total")
gpu_hit, t_gpu = chat(target)

same = cold == cpu_hit == gpu_hit
print(f"cold      ttft {t_cold:.3f}s")
print(f"cpu hit   ttft {t_cpu:.3f}s  (host-tier tokens +{host_after - host_before:.0f})")
print(f"gpu hit   ttft {t_gpu:.3f}s")
print(f"byte-identical: {same}")
if not same:
    print("cold   :", repr(cold))
    print("cpu hit:", repr(cpu_hit))
    print("gpu hit:", repr(gpu_hit))
sys.exit(0 if same and host_after > host_before else 1)
