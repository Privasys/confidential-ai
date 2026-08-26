package models

import (
	"fmt"
	"os"
)

// Dirty-load JIT/compile cache invalidation.
//
// The Triton kernel cache (TRITON_CACHE_DIR), vLLM's torch.compile cache
// (VLLM_CACHE_ROOT) and FlashInfer's cache (via HOME) all persist under
// /data/.cache so warm restarts skip minutes of JIT compilation. The flip
// side: a load killed mid-JIT-write (supersede, OOM, crash) leaves
// truncated entries behind, and a poisoned cache hangs the next engine
// bring-up — observed on m4 (2026-08-26) as an indefinite CUDA-graph
// capture hang on every model, while the identical engine on a cache-less
// app captured fine.
//
// Mechanism: a per-task sentinel file marks a load in flight. Finding the
// sentinel already present at the next load start means the previous load
// of this task never reached ready, so every JIT cache dir is wiped before
// vLLM starts. The sentinel is per task so a generate load in flight never
// looks dirty to a concurrent embed/rerank load (their JIT writes are
// active, not stale); the wipe itself is global because the cache dirs are
// shared. entrypoint.sh separately wipes on image-digest change
// (cross-version invalidation) — together they bound cache poison to "the
// exact build that wrote it, and only while its load chain stays healthy".

const cacheRoot = "/data/.cache"

// jitCacheDirs are the shared JIT/compile cache locations under cacheRoot.
var jitCacheDirs = []string{"triton", "vllm", "torch", "flashinfer"}

func (m *Manager) dirtyLoadSentinelPath() string {
	return fmt.Sprintf("%s/.load-in-progress-%s", cacheRoot, m.task)
}

// wipeCachesIfDirtyLoad wipes the JIT caches when this task's previous
// load never reached ready, then (re)arms the sentinel for the load that
// is about to start. Best-effort: on a storage-less app the paths live in
// the container overlay and every step degrades to a no-op.
func (m *Manager) wipeCachesIfDirtyLoad() {
	sentinel := m.dirtyLoadSentinelPath()
	if _, err := os.Stat(sentinel); err == nil {
		fmt.Fprintf(os.Stderr,
			"{\"level\":\"warn\",\"msg\":\"previous load never reached ready; wiping JIT caches\",\"task\":%q}\n",
			m.task)
		for _, d := range jitCacheDirs {
			os.RemoveAll(cacheRoot + "/" + d)
		}
	}
	if err := os.MkdirAll(cacheRoot, 0o755); err != nil {
		return
	}
	f, err := os.Create(sentinel)
	if err != nil {
		return
	}
	f.Close()
}

// clearDirtyLoadSentinel marks this task's load as having reached ready:
// its JIT cache writes are complete and consistent.
func (m *Manager) clearDirtyLoadSentinel() {
	os.Remove(m.dirtyLoadSentinelPath())
}
