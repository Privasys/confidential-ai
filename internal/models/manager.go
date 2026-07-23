// Package models manages the vLLM subprocess lifecycle for dynamic model
// loading and unloading. Instead of starting vLLM at container boot, the
// Go proxy starts immediately and vLLM is started on-demand when
// POST /v1/models/load is called.
package models

import (
	"bufio"
	"context"
	"crypto/sha256"
	"encoding/hex"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
	"syscall"
	"time"
)

// State represents the model manager state machine.
type State string

const (
	StateIdle    State = "idle"
	StateLoading State = "loading"
	StateReady   State = "ready"
	StateFailed  State = "failed"
)

// Task selects the vLLM runner mode for an instance. The fleet serves the
// chat LLM (generate) next to the small pooling models Drive's RAG needs
// (embed + rerank) on the same GPU, each as its own vLLM subprocess.
type Task string

const (
	TaskGenerate Task = "generate"
	TaskEmbed    Task = "embed"
	TaskRerank   Task = "rerank"
)

// NormalizeTask maps the wire value to a known Task ("" means generate,
// the historic single-model behaviour). Unknown values return an error.
func NormalizeTask(t Task) (Task, error) {
	switch t {
	case "", TaskGenerate:
		return TaskGenerate, nil
	case TaskEmbed, TaskRerank:
		return t, nil
	default:
		return "", fmt.Errorf("unknown task %q (want generate, embed or rerank)", t)
	}
}

// LoadRequest is the body of POST /v1/models/load.
//
// `Model` is the canonical served name. It is what vLLM will register
// (`--served-model-name`), what `GET /v1/models` reports, and what every
// chat-completions request must put in its `model` field. The proxy never
// rewrites client-supplied names; the orchestrator is expected to publish
// the same string in its instance discovery API.
//
// `Source` is an optional loader hint: a filesystem path or a HuggingFace
// repo. When empty it defaults to `Model` resolved against modelsDir.
// This split lets the served name stay friendly ("gemma4-31b") even when
// the on-disk path is awkward ("/models/gemma-4-31b-it").
type LoadRequest struct {
	// Task selects the vLLM runner mode. Empty means generate (the chat
	// LLM). embed serves an OpenAI-compatible /v1/embeddings pooling
	// model; rerank serves a sequence-classification scorer behind
	// /v1/rerank. Each task runs as its own vLLM subprocess on its own
	// port; see Fleet.
	Task Task `json:"task,omitempty"`

	Model                string  `json:"model"`                            // canonical served name (required)
	Source               string  `json:"source,omitempty"`                 // optional loader hint: path or HF repo
	Dtype                string  `json:"dtype,omitempty"`                  // default "auto"
	Quantization         string  `json:"quantization,omitempty"`           // awq, gptq, int4, etc.
	MaxModelLen          int     `json:"max_model_len,omitempty"`          // default 8192
	GPUMemoryUtilization float64 `json:"gpu_memory_utilization,omitempty"` // default 0.90

	// ReasoningParser, ToolCallParser, ChatTemplate and EnableAutoToolChoice
	// wire vLLM's per-architecture reasoning + tool-call parsers. When
	// ReasoningParser is set the OpenAI streaming response carries
	// `delta.reasoning_content` separately from `delta.content`, which the
	// chat front-end re-wraps in <think>…</think> sentinels so existing
	// reasoning UI keeps working without the model having to invent any
	// markup itself. ChatTemplate, when non-empty, is passed verbatim to
	// `--chat-template`; the entrypoint downloads the official Gemma 4
	// template into /opt/vllm-templates so callers can reference
	// `gemma4` (canonical) without needing the absolute path.
	ReasoningParser      string `json:"reasoning_parser,omitempty"`
	ToolCallParser       string `json:"tool_call_parser,omitempty"`
	EnableAutoToolChoice bool   `json:"enable_auto_tool_choice,omitempty"`
	ChatTemplate         string `json:"chat_template,omitempty"`
	// EnableThinking, when true, sets vLLM's
	// `--default-chat-template-kwargs '{"enable_thinking": true}'` so
	// reasoning is on for every request unless the caller opts out via
	// `chat_template_kwargs`. Required by the Gemma 4 thinking recipe.
	EnableThinking bool `json:"enable_thinking,omitempty"`

	// MaxNumSeqs caps vLLM's `--max-num-seqs`. Required for
	// linear-attention / Mamba-cache models (e.g. Qwen3.6-35B-A3B with
	// Gated DeltaNet) where each in-flight sequence consumes one Mamba
	// cache block: vLLM aborts CUDA graph capture with
	// `max_num_seqs exceeds available Mamba cache blocks` when the
	// default (1024) is too large for the GPU. 0 means vLLM default.
	MaxNumSeqs int `json:"max_num_seqs,omitempty"`

	// EnableChunkedPrefill switches the prefill policy. Default (false)
	// keeps the single-block reproducibility recipe:
	// `--no-enable-chunked-prefill` with `--max-num-batched-tokens`
	// forced to at least max_model_len so any prompt is one prefill
	// step. That recipe scales the startup profile_run's activation
	// reservation with max_model_len, which makes long-context
	// deployments (>=128k) waste 15-25 GiB; for those, set this true
	// and a bounded MaxNumBatchedTokens (default 32768 when chunked).
	// Chunk boundaries change the FP reduction order, so record the
	// choice in the reproducibility config — replay must use the same
	// policy. See the vLLM memory-tuning notes.
	EnableChunkedPrefill bool `json:"enable_chunked_prefill,omitempty"`

	// MaxNumBatchedTokens overrides `--max-num-batched-tokens`.
	// 0 means: max(16384, max_model_len) when chunked prefill is off
	// (the single-block invariant), 32768 when it is on. When chunked
	// prefill is off the value is clamped up to max_model_len because
	// vLLM refuses MNBT < max_model_len in that mode.
	MaxNumBatchedTokens int `json:"max_num_batched_tokens,omitempty"`

	// EnablePrefixCaching turns on vLLM automatic prefix caching
	// (`--enable-prefix-caching`). Default false — matching vLLM's own
	// default for the hybrid (Mamba) models we serve, where support is
	// still experimental (mamba_cache_mode=align; hits only on fully
	// completed ~528-token aligned blocks). Reuse is scoped per caller
	// by the proxy's cache_salt injection (see handler/cache_salt.go),
	// and every response disclosed its hits via
	// reproducibility.cached_tokens, so enabling this never silently
	// weakens the replay contract — it is recorded per request.
	EnablePrefixCaching bool `json:"enable_prefix_caching,omitempty"`

	// KVCacheDtype sets `--kv-cache-dtype` (e.g. "fp8"). Halves the
	// attention KV footprint on the hybrid models (20 -> 10 KiB/token
	// on Qwen3.6-35B). Changes outputs vs fp16 KV, so it is part of
	// the attested config. Empty means vLLM default ("auto").
	KVCacheDtype string `json:"kv_cache_dtype,omitempty"`

	// MambaSSMCacheDtype sets `--mamba-ssm-cache-dtype` for the Gated
	// DeltaNet recurrent state (~31 MiB/seq bf16, double at float32).
	// Pin it explicitly so a vLLM default change can never silently
	// halve the available sequence slots. Empty means vLLM default.
	MambaSSMCacheDtype string `json:"mamba_ssm_cache_dtype,omitempty"`

	// MaxCudagraphCaptureSize caps `--max-cudagraph-capture-size`
	// (vLLM default 512). Set it to MaxNumSeqs on memory-tight
	// deployments — capturing graphs for batch sizes the scheduler
	// will never admit only burns VRAM. 0 means vLLM default.
	MaxCudagraphCaptureSize int `json:"max_cudagraph_capture_size,omitempty"`

	// HFOverrides is a JSON object passed verbatim to `--hf-overrides`.
	// Auto-filled for known model families (the Qwen3 reranker must be
	// served as sequence classification: architectures
	// Qwen3ForSequenceClassification, classifier_from_token [no,yes],
	// is_original_qwen3_reranker); explicit values win, as with the
	// parser auto-recipes.
	HFOverrides string `json:"hf_overrides,omitempty"`
}

// Status is the response for GET /v1/models/status.
type Status struct {
	State       State   `json:"state"`
	Task        Task    `json:"task,omitempty"`
	Model       string  `json:"model,omitempty"`
	ModelDigest string  `json:"model_digest,omitempty"`
	Progress    float64 `json:"progress,omitempty"`    // 0.0 - 1.0 during loading
	Message     string  `json:"message,omitempty"`     // human-readable status
	ElapsedSec  float64 `json:"elapsed_s,omitempty"`
	Error       string  `json:"error,omitempty"`
	// StderrTail is the recent vLLM stderr (up to the last 40 lines), surfaced
	// ONLY on failure. The Error field carries a single best-guess line; the real
	// root cause of a startup crash ("...See root cause above") is usually an
	// earlier line, so an owner debugging a failed load needs the whole tail —
	// production enclaves have no shell or journal to read it any other way.
	StderrTail []string `json:"stderr_tail,omitempty"`
}

// Manager manages one vLLM subprocess lifecycle (one task, one port).
type Manager struct {
	task        Task   // runner mode this instance serves
	modelsDir   string // path to model directory (e.g. /models)
	vllmPort    int    // port for vLLM to listen on
	roothashDir string // directory of <model>.roothash files written by disk-mounter

	// stateFile, when non-empty, is the path on the per-container
	// encrypted volume (typically /data/last-load.json) where the
	// most-recently-successful LoadRequest is persisted. RestoreFromDisk
	// reads it on boot and re-issues Load so a container restart (e.g.
	// post-VM-reboot) auto-recovers the previously-served model without
	// needing the orchestrator to issue a fresh /v1/models/load.
	stateFile string

	mu           sync.RWMutex
	state        State
	model        string
	modelDigest  string
	quantization string
	progress     float64
	message      string
	loadStart    time.Time
	loadErr      string
	cmd          *exec.Cmd
	cancel       context.CancelFunc

	// gen is the load generation. Every Load/Unload bumps it, and every
	// state write from a load's goroutines (doLoad, runVLLM, the two
	// parseProgress scanners, the crash watcher) is guarded on it, so a
	// SUPERSEDED load can never clobber the state of its successor — the
	// failure mode behind the 2026-07-23 dev incident where a dying
	// superseded load marked the replacement load failed.
	gen uint64

	// procDone closes when runVLLM's single cmd.Wait owner returns.
	// Unload waits on it instead of calling cmd.Wait() itself: two
	// concurrent Waits on one exec.Cmd are a data race.
	procDone chan struct{}

	// appliedReq is the effective (defaults-applied) LoadRequest of the
	// load currently serving or in flight. Load compares against it so a
	// re-delivered identical configuration stays a no-op while a changed
	// parameter set triggers a reload.
	appliedReq LoadRequest

	// stderrTail is a small ring of the most recent vLLM stderr lines,
	// kept so a process death can surface the ACTUAL error (argparse
	// failure, ValueError, OOM traceback) in the status document instead
	// of a bare exit code.
	stderrTail []string
}

// NewManager creates a new model manager for one task. stateFile may be
// empty to disable persistence (in-process testing). When non-empty the
// file is written after every successful Load and read by RestoreFromDisk.
func NewManager(task Task, modelsDir string, vllmPort int, roothashDir, stateFile string) *Manager {
	if task == "" {
		task = TaskGenerate
	}
	return &Manager{
		task:        task,
		modelsDir:   modelsDir,
		vllmPort:    vllmPort,
		roothashDir: roothashDir,
		stateFile:   stateFile,
		state:       StateIdle,
	}
}

// Task returns the runner mode this instance serves.
func (m *Manager) Task() Task { return m.task }

// Upstream returns the base URL of this instance's vLLM server.
func (m *Manager) Upstream() string {
	// 127.0.0.1, NOT localhost: the container /etc/hosts on the per-
	// container network stack has no localhost entry, and Go resolves
	// "localhost" via DNS → NXDOMAIN (found live on m5-dev-ai: every
	// readiness poll failed against a serving vLLM for hours).
	return fmt.Sprintf("http://127.0.0.1:%d", m.vllmPort)
}

// Status returns the current model manager status.
func (m *Manager) Status() Status {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s := Status{
		State:       m.state,
		Task:        m.task,
		Model:       m.model,
		ModelDigest: m.modelDigest,
	}

	switch m.state {
	case StateLoading:
		s.Progress = m.progress
		s.Message = m.message
		s.ElapsedSec = time.Since(m.loadStart).Seconds()
	case StateFailed:
		s.Error = m.loadErr
		if !m.loadStart.IsZero() {
			s.ElapsedSec = time.Since(m.loadStart).Seconds()
		}
		// Surface the full stderr ring so the actual root cause is readable via
		// the API (copied under the lock we already hold — do not call
		// stderrTailSuffix here, it re-acquires m.mu).
		if len(m.stderrTail) > 0 {
			s.StderrTail = append([]string(nil), m.stderrTail...)
		}
	case StateReady:
		s.Message = "Model loaded and serving"
	}

	return s
}

// IsReady returns true if a model is loaded and serving.
func (m *Manager) IsReady() bool {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.state == StateReady
}

// ModelName returns the current model name.
func (m *Manager) ModelName() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.model
}

// ModelDigest returns the current model digest.
func (m *Manager) ModelDigest() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.modelDigest
}

// Quantization returns the current quantization setting.
func (m *Manager) Quantization() string {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.quantization
}

// normalizeRequest produces the EFFECTIVE request for this instance:
// task stamp, sizing defaults, and the family recipes (reasoning/tool
// parsers, reranker hf_overrides). It is the ONLY place a request is
// mutated, and it runs BEFORE the idempotency compare, so the compared,
// served, and persisted requests are one and the same value. The
// recipes previously applied later (in doLoad, after appliedReq was
// stored) — a persisted request then carried recipe fields a fresh wire
// request lacked, every restore→re-configure compared unequal, and a
// byte-identical owner configure spuriously unloaded a serving 35B
// (the 2026-07-23 dev incident). Idempotent: normalize(normalize(x)) ==
// normalize(x), so restoring an old persisted (recipe-carrying) file
// converges to the same effective request.
func (m *Manager) normalizeRequest(req LoadRequest) LoadRequest {
	// The instance's task is fixed at construction; the request either
	// matches or is silently stamped (callers route by task already).
	req.Task = m.task

	if req.Dtype == "" {
		req.Dtype = "auto"
	}
	if m.task == TaskEmbed || m.task == TaskRerank {
		// Pooling / classification instances share the GPU with the main
		// LLM: a small utilisation slice with full 32k document context,
		// and a tight batch cap so embedding bursts don't add decode
		// jitter to the chat model (CC mode has no MPS; the CUDA contexts
		// time-slice). 0.08 measured on H100-80GB: ~1.2 GiB bf16 weights
		// + ~1.2 GiB CUDA/activations + the 3.5 GiB KV floor vLLM demands
		// for one 32k sequence (util 0.05 left only 1.68 GiB KV and the
		// engine refused to start). See vllm-memory-tuning.md, ai-plan
		// §7.5.
		if req.MaxModelLen == 0 {
			req.MaxModelLen = 32768
		}
		if req.GPUMemoryUtilization == 0 {
			req.GPUMemoryUtilization = 0.08
		}
		if req.MaxNumSeqs == 0 {
			req.MaxNumSeqs = 4
		}
		// The Qwen3 reranker must be served as sequence classification
		// with the yes-logit as the score (see doLoad's previous home of
		// this rule for the full rationale).
		if m.task == TaskRerank && req.HFOverrides == "" &&
			strings.Contains(strings.ToLower(req.Model), "qwen3-reranker") {
			req.HFOverrides = `{"architectures": ["Qwen3ForSequenceClassification"], "classifier_from_token": ["no", "yes"], "is_original_qwen3_reranker": true}`
		}
		return req
	}

	if req.MaxModelLen == 0 {
		req.MaxModelLen = 8192
	}
	if req.GPUMemoryUtilization == 0 {
		req.GPUMemoryUtilization = 0.90
	}
	if req.MaxNumSeqs == 0 {
		// vLLM's own default (1024) is sized for serving clusters and
		// makes hybrid-attention models (Qwen 3.6 MoE: one Mamba cache
		// block per decode sequence) fail at KV-cache init on a single
		// H100. NOTE: for such models a high value can still abort CUDA
		// graph capture ("max_num_seqs exceeds available Mamba cache
		// blocks") at long contexts — cap max_model_len accordingly, or
		// lower this via the load request's max_num_seqs.
		req.MaxNumSeqs = 256
	}

	// Family recipes for the chat model (explicit fields always win; see
	// the per-family notes that used to live in doLoad).
	lname := strings.ToLower(req.Model)
	if strings.Contains(lname, "gemma4") {
		if req.ReasoningParser == "" {
			req.ReasoningParser = "gemma4"
		}
		if req.ToolCallParser == "" {
			req.ToolCallParser = "gemma4"
			req.EnableAutoToolChoice = true
		}
		if req.ChatTemplate == "" {
			req.ChatTemplate = "gemma4"
		}
		if !req.EnableThinking {
			req.EnableThinking = true
		}
	}
	if strings.Contains(lname, "qwen3") || strings.Contains(lname, "qwen36") || strings.Contains(lname, "qwen35") {
		if req.ReasoningParser == "" {
			req.ReasoningParser = "qwen3"
		}
		if req.ToolCallParser == "" {
			req.ToolCallParser = "qwen3_coder"
			req.EnableAutoToolChoice = true
		}
		if !req.EnableThinking {
			req.EnableThinking = true
		}
	}
	return req
}

// Load starts loading a model. Idempotent on the FULL effective request:
// re-delivering the identical configuration (e.g. the orchestrator
// re-configuring after a restart) is a no-op whether the load is serving
// OR still in flight; a request with different parameters supersedes —
// the serving/in-flight load is stopped and the new one started.
// Matching by model name alone silently discarded parameter changes
// (applying max_model_len=262144 over a running 8192 instance did
// nothing, and tool-augmented prompts then blew the stale context
// window).
func (m *Manager) Load(req LoadRequest) error {
	req = m.normalizeRequest(req)

	m.mu.Lock()

	// Idempotent: the identical effective configuration is already
	// serving, or already on its way (join the in-flight load rather
	// than erroring — a re-issued configure during startup used to get
	// "already loading" and surface as a spurious failure).
	if (m.state == StateReady || m.state == StateLoading) && req == m.appliedReq {
		m.mu.Unlock()
		return nil
	}

	// A different configuration is serving or in flight: supersede it.
	// Unload bumps the generation first, so the dying load's goroutines
	// can no longer touch state, then kills the process group.
	if m.state == StateReady || m.state == StateLoading {
		m.mu.Unlock()
		if err := m.Unload(); err != nil {
			return fmt.Errorf("unload current model: %w", err)
		}
		m.mu.Lock()
	}

	m.gen++
	gen := m.gen
	m.state = StateLoading
	m.model = req.Model
	m.quantization = req.Quantization
	m.appliedReq = req
	m.modelDigest = ""
	m.progress = 0
	m.message = "Starting vLLM..."
	m.loadStart = time.Now()
	m.loadErr = ""
	m.stderrTail = nil
	m.mu.Unlock()

	go m.doLoad(req, gen)
	return nil
}

// ifGen runs fn under the lock only when gen is still the current load
// generation. Every state write from a load's goroutines goes through
// this so a superseded load cannot clobber its successor.
func (m *Manager) ifGen(gen uint64, fn func()) bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.gen != gen {
		return false
	}
	fn()
	return true
}

// Unload stops vLLM and frees resources.
//
// Deadlock guard: we grab the cmd/cancel handles under the lock, then
// release the lock BEFORE killing + waiting. Holding m.mu across
// cmd.Wait() was the root cause of the "manager hangs on unload" bug —
// every other handler (Status, Load, etc.) blocks on RLock/Lock while
// Wait is waiting for vLLM's grandchildren to release the stderr pipe,
// which can take indefinitely if any worker process becomes a zombie
// reparented to PID 1.
//
// The kill also targets the whole process group (negative PID) so vLLM
// worker subprocesses (EngineCore, ray, mp spawn) die together; combined
// with cmd.WaitDelay (set in doLoad) this guarantees Wait returns even
// if a grandchild keeps holding the pipe.
func (m *Manager) Unload() error {
	m.mu.Lock()
	if m.state == StateIdle {
		m.mu.Unlock()
		return nil
	}
	// Invalidate the load generation BEFORE killing: any in-flight load
	// goroutine (doLoad/runVLLM/scanners) checks the generation under
	// this same lock before writing state, so after this point the dying
	// load is a pure bystander — it can neither mark the instance failed
	// nor persist its request.
	m.gen++
	cancel := m.cancel
	cmd := m.cmd
	procDone := m.procDone
	m.cancel = nil
	m.mu.Unlock()

	if cancel != nil {
		cancel()
	}
	if cmd != nil && cmd.Process != nil {
		// Kill the entire process group; vLLM forks worker processes
		// that inherit the stderr pipe. SIGTERM first, give them 2 s
		// to flush, then SIGKILL.
		pgid, err := syscall.Getpgid(cmd.Process.Pid)
		if err == nil {
			_ = syscall.Kill(-pgid, syscall.SIGTERM)
			time.Sleep(2 * time.Second)
			_ = syscall.Kill(-pgid, syscall.SIGKILL)
		} else {
			_ = cmd.Process.Kill()
		}
		// Wait for the reap via runVLLM's single Wait owner — calling
		// cmd.Wait() here as well is a data race on exec.Cmd (two
		// concurrent Waits; caught by -race once Load learned to
		// supersede in-flight loads). procDone closes when that owner's
		// Wait returns; cmd.WaitDelay bounds it at 10 s after the kill
		// even if a grandchild keeps the pipe open.
		if procDone != nil {
			select {
			case <-procDone:
			case <-time.After(15 * time.Second):
			}
		}
	}

	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = StateIdle
	m.model = ""
	m.modelDigest = ""
	m.quantization = ""
	m.progress = 0
	m.message = ""
	m.loadErr = ""
	m.cmd = nil
	m.procDone = nil
	return nil
}

// ListAvailable returns model directories found under modelsDir.
func (m *Manager) ListAvailable() ([]string, error) {
	entries, err := os.ReadDir(m.modelsDir)
	if err != nil {
		return nil, err
	}
	var models []string
	for _, e := range entries {
		if e.IsDir() {
			models = append(models, e.Name())
		}
	}
	return models, nil
}

// doLoad runs the vLLM subprocess and tracks its progress. gen is the
// load generation captured by Load; every state write is guarded on it.
func (m *Manager) doLoad(req LoadRequest, gen uint64) {
	ctx, cancel := context.WithCancel(context.Background())
	// Register the cancel handle only while this load is still current:
	// Unload bumps the generation under the same lock before reading the
	// handle, so either it sees our cancel (and cancels us) or we see the
	// stale generation here and abort before spawning anything.
	if !m.ifGen(gen, func() { m.cancel = cancel }) {
		cancel()
		return
	}

	defer func() {
		cancel()
		m.ifGen(gen, func() { m.cancel = nil })
	}()

	// Resolve model path. Source overrides Model when present, so the
	// served name (the canonical id reported back to clients) can be
	// short and friendly while the loader still finds the on-disk
	// safetensors directory. Family recipes (reasoning/tool parsers,
	// reranker hf_overrides) were already applied by normalizeRequest in
	// Load — BEFORE the idempotency compare and persistence, so the
	// compared, served, and persisted requests are identical.
	loaderID := req.Source
	if loaderID == "" {
		loaderID = req.Model
	}
	modelPath := m.resolveModelPath(loaderID)

	m.runVLLM(ctx, req, loaderID, modelPath, gen)
}

// runVLLM spawns the vLLM subprocess for the (normalized) request,
// tracks progress, and blocks until the process exits. Shared tail of
// doLoad for all tasks; ctx is doLoad's cancellable load context, gen
// the load generation guarding every state write.
func (m *Manager) runVLLM(ctx context.Context, req LoadRequest, loaderID, modelPath string, gen uint64) {
	args := buildVLLMArgs(req, modelPath, m.vllmPort)

	cmd := exec.CommandContext(ctx, "vllm", args...)
	cmd.Env = append(os.Environ(),
		// Force PyTorch / cuBLAS into deterministic-workspace mode.
		// Required for torch.use_deterministic_algorithms() under
		// CUDA >= 10.2; vLLM picks this up via PyTorch.
		"CUBLAS_WORKSPACE_CONFIG=:4096:8",
		"PYTHONHASHSEED=0",
		// VLLM_USE_V1 intentionally NOT set: defaults to V1 in
		// vLLM >= 0.19, which is what we want for throughput.
		//
		// Bound the FlashInfer/Triton JIT compile fan-out. Uncapped, the
		// MoE kernel build launches one cicc per core (26 on the H100
		// shape) at 3-6 GB each — enough to stall the whole host in
		// reclaim (or, under a memcg, OOM it). MAX_JOBS gates ninja /
		// torch cpp_extension builds; NVCC_THREADS gates per-invocation
		// nvcc parallelism; FLASHINFER_JIT_MAX_WORKERS gates FlashInfer's
		// own pool on versions that support it.
		"MAX_JOBS=6",
		"NVCC_THREADS=2",
		"FLASHINFER_JIT_MAX_WORKERS=6",
	)
	// Persist the JIT/compile caches across container recreations by
	// homing vLLM on the encrypted volume when one is mounted. FlashInfer
	// resolves its cache via expanduser("~/.cache/flashinfer") — HOME, not
	// XDG — and a fresh container overlay otherwise pays the full
	// per-architecture nvcc compile (~8 min for the Qwen GDN sm_90a
	// kernels) on EVERY image upgrade/redeploy. Triton and HF caches ride
	// along, all inside the LUKS volume, never on the host.
	if info, statErr := os.Stat("/data"); statErr == nil && info.IsDir() {
		cmd.Env = append(cmd.Env, "HOME=/data")
	}
	// Put vLLM + every worker it spawns in their own process group so
	// Unload() can SIGKILL the whole tree at once via `kill -- -pgid`.
	// Without this, vLLM EngineCore / ray workers survive Process.Kill
	// on the main pid and keep the stderr pipe open, so cmd.Wait()
	// blocks forever (the original "manager hangs on unload" bug).
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	// Bound cmd.Wait() at 10 s after the process exits even if a
	// grandchild still holds an inherited pipe fd. Go's exec package
	// forcibly closes the pipes once WaitDelay elapses.
	cmd.WaitDelay = 10 * time.Second

	// Pipe stderr for progress tracking.
	stderr, err := cmd.StderrPipe()
	if err != nil {
		m.setFailed(gen, "failed to create stderr pipe: "+err.Error())
		return
	}
	// vLLM stdout is drained into the same ring as stderr — NEVER wired to
	// our own stdout. The process stdout is the container's log FIFO; the
	// 35B's startup flood can fill it and block every writer in the
	// container (see parseProgress).
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		m.setFailed(gen, "failed to create stdout pipe: "+err.Error())
		return
	}

	// Register the cmd handle only while still the current load —
	// superseded before Start means nothing was spawned and nothing to
	// kill; just walk away.
	if !m.ifGen(gen, func() {
		m.cmd = cmd
		m.message = "Starting vLLM process..."
	}) {
		return
	}

	if err := cmd.Start(); err != nil {
		m.setFailed(gen, "failed to start vLLM: "+err.Error())
		return
	}

	// Parse both streams for progress/tail in background. parseProgress
	// appends to the shared ring under m.mu, so two scanners are safe;
	// both carry gen so a superseded load's dying output never pollutes
	// the successor's tail/progress.
	go m.parseProgress(stderr, gen)
	go m.parseProgress(stdout, gen)

	// Single owner of cmd.Wait: every consumer below reads waitCh, so the
	// exit status is observed exactly once and the readiness loop can
	// fail FAST on process death instead of waiting out its cap. Buffered
	// so the goroutine never leaks if nobody reads. procDone additionally
	// broadcasts the reap to Unload, which must never call cmd.Wait()
	// itself (two concurrent Waits on one exec.Cmd are a data race).
	waitCh := make(chan error, 1)
	procDone := make(chan struct{})
	go func() {
		err := cmd.Wait()
		waitCh <- err
		close(procDone)
	}()
	m.ifGen(gen, func() { m.procDone = procDone })

	// Wait for vLLM to be ready OR for the process to die.
	if err := m.waitForReady(ctx, waitCh, gen); err != nil {
		if exitErr, died := err.(*vllmExitError); died {
			// The process died: fail immediately with the REAL reason
			// (last error lines off stderr), not a generic timeout N
			// minutes later. On 2026-07-15 every engine death (argparse
			// error, KV-floor ValueError, memcg OOM kill) hid behind
			// "not ready after N minutes" for its full window.
			m.setFailed(gen, exitErr.Error())
		} else {
			m.setFailed(gen, "vLLM health check failed: "+err.Error())
			// Kill the whole process GROUP, exactly like Unload: killing
			// only the APIServer pid leaks its EngineCore children, which
			// keep their full GPU allocation alive. Three leaked cores held
			// 77.6/81.5 GiB on m5-dev-ai (2026-07-15) and starved every
			// subsequent load into the same timeout — a leak loop.
			if pgid, perr := syscall.Getpgid(cmd.Process.Pid); perr == nil {
				_ = syscall.Kill(-pgid, syscall.SIGTERM)
				time.Sleep(2 * time.Second)
				_ = syscall.Kill(-pgid, syscall.SIGKILL)
			} else {
				_ = cmd.Process.Kill()
			}
			<-waitCh
		}
		return
	}

	// Compute model digest. Prefer the dm-verity root hash from the
	// model disk (covers every byte the kernel will ever serve) over
	// the legacy index-file hash (covers only the index json). The
	// roothash filename is keyed off the loader id (Source or Model),
	// not the served name, because that is what the disk-mounter
	// actually mounts.
	digest := m.lookupVerityRoothash(loaderID)
	if digest == "" {
		digest = m.computeDigest(modelPath)
	}

	if !m.ifGen(gen, func() {
		m.state = StateReady
		m.model = req.Model
		m.modelDigest = digest
		m.progress = 1.0
		m.message = "Model loaded and serving"
	}) {
		// Superseded while becoming ready: Unload already killed the
		// process group; do not persist the dead configuration.
		return
	}

	// Persist the successful load so a container restart auto-recovers.
	m.persistRequest(req)

	// Wait for process exit (blocks until vLLM dies or is killed).
	if err := <-waitCh; err != nil {
		// gen guard: after an Unload/supersede this crash belongs to the
		// old load and must not fail the new one.
		m.ifGen(gen, func() {
			if m.state == StateReady {
				// Unexpected crash: surface the stderr tail, not just the
				// exit code.
				m.state = StateFailed
				m.loadErr = "vLLM process exited unexpectedly: " + err.Error() + m.stderrTailSuffix()
			}
		})
	}
}

// vllmExitError marks a readiness failure caused by the vLLM process
// dying (as opposed to a health-check timeout). It carries the exit
// status plus the most informative recent stderr lines so the operator
// sees the actual Python/vLLM error in the status document.
type vllmExitError struct {
	waitErr error
	tail    string
}

func (e *vllmExitError) Error() string {
	msg := "vLLM exited during startup"
	if e.waitErr != nil {
		msg += " (" + e.waitErr.Error() + ")"
	}
	return msg + e.tail
}

// buildVLLMArgs translates a LoadRequest (with defaults already
// applied) into the `vllm serve` argv. Split out of doLoad so the flag
// policy is unit-testable.
//
// Reproducibility tradeoffs we accept:
//
//   - CUDA graphs ENABLED (no --enforce-eager). Graph capture +
//     replay is itself deterministic: same kernel DAG, same memory
//     offsets, same streams. Eager mode was costing ~30x throughput
//     for no determinism gain in our locked environment (pinned
//     kernel, NVIDIA driver, CUDA, vLLM version, model digest).
//
//   - V1 engine ENABLED (no VLLM_USE_V1=0 env). The legacy V0
//     scheduler underutilises the H100 tensor cores.
//
//   - Prefill policy is per-deployment. Default (EnableChunkedPrefill
//     false): chunked prefill OFF and --max-num-batched-tokens forced
//     to max(16384, max_model_len) so any prompt up to max_model_len
//     is one prefill step — chunk boundaries would change the order
//     of FP reductions. The cost is that vLLM's startup profile_run
//     does a dummy forward of MNBT tokens, so the activation
//     reservation scales with max_model_len; at >=128k contexts that
//     wastes 15-25 GiB that the KV + Mamba caches need. Long-context
//     deployments set EnableChunkedPrefill with a bounded MNBT
//     (default 32768) and record the policy in the attested config —
//     replay must use the same chunking schedule. Note MNBT larger
//     than necessary buys nothing and ONLY inflates profile_run
//     (an earlier 2*max_model_len setting halved the achievable
//     context before that was understood).
//
// What none of this guarantees: batch-invariance. With continuous
// batching, two concurrent requests may see different reductions
// across the batch dimension. Determinism holds per-request when
// traffic is serialised; true concurrent determinism needs
// batch-invariant kernels, which upstream does not yet support for
// GDN models (vllm#42960).
func buildVLLMArgs(req LoadRequest, modelPath string, port int) []string {
	// Use a short, stable served-model-name. If the caller passed an
	// absolute filesystem path (e.g. "/models/qwen36-35b-a3b-fp8"),
	// vLLM would otherwise advertise the full path as the model id,
	// which doesn't match what UI clients send and produces a 404 from
	// vLLM ("model X does not exist"). Strip to basename so the served
	// name matches the directory name under modelsDir.
	servedName := req.Model
	if filepath.IsAbs(servedName) {
		servedName = filepath.Base(servedName)
	}

	args := []string{
		"serve", modelPath,
		"--served-model-name", servedName,
		"--seed", "0",
		"--tensor-parallel-size", "1",
	}

	// Runner mode. generate is vLLM's default; the pooling instances
	// select their runner explicitly. NB vLLM 0.22 removed the legacy
	// `--task` flag ("unrecognized arguments: --task embed", hit live on
	// m5-dev-ai): the modern selection is `--runner pooling` plus a
	// `--convert` adapter — `embed` serves the OpenAI-compatible
	// /v1/embeddings pooling API, `classify` the cross-encoder scoring
	// API behind /v1/rerank (the Qwen3 reranker is loaded as sequence
	// classification via HFOverrides and scores with the yes-logit).
	switch req.Task {
	case TaskEmbed:
		args = append(args, "--runner", "pooling", "--convert", "embed")
	case TaskRerank:
		args = append(args, "--runner", "pooling", "--convert", "classify")
	}
	if req.HFOverrides != "" {
		args = append(args, "--hf-overrides", req.HFOverrides)
	}

	batchedTokens := req.MaxNumBatchedTokens
	if req.EnableChunkedPrefill {
		if batchedTokens == 0 {
			batchedTokens = 32768
		}
		args = append(args, "--enable-chunked-prefill")
	} else {
		// Single-block prefill: vLLM rejects MNBT < max_model_len in
		// this mode, so clamp up rather than fail at engine init.
		if batchedTokens < req.MaxModelLen {
			batchedTokens = req.MaxModelLen
		}
		if batchedTokens < 16384 {
			batchedTokens = 16384
		}
		args = append(args, "--no-enable-chunked-prefill")
	}

	args = append(args,
		"--max-num-batched-tokens", fmt.Sprintf("%d", batchedTokens),
		"--no-enable-log-requests",
		"--max-model-len", fmt.Sprintf("%d", req.MaxModelLen),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", req.GPUMemoryUtilization),
		"--dtype", req.Dtype,
		"--port", fmt.Sprintf("%d", port),
	)
	if req.MaxNumSeqs > 0 {
		args = append(args, "--max-num-seqs", fmt.Sprintf("%d", req.MaxNumSeqs))
	}
	if req.EnablePrefixCaching {
		args = append(args, "--enable-prefix-caching")
	}
	if req.KVCacheDtype != "" {
		args = append(args, "--kv-cache-dtype", req.KVCacheDtype)
	}
	if req.MambaSSMCacheDtype != "" {
		args = append(args, "--mamba-ssm-cache-dtype", req.MambaSSMCacheDtype)
	}
	if req.MaxCudagraphCaptureSize > 0 {
		args = append(args, "--max-cudagraph-capture-size", fmt.Sprintf("%d", req.MaxCudagraphCaptureSize))
	}
	if req.Quantization != "" && req.Quantization != "none" {
		args = append(args, "--quantization", req.Quantization)
	}
	if req.ReasoningParser != "" {
		args = append(args, "--reasoning-parser", req.ReasoningParser)
	}
	if req.ToolCallParser != "" {
		args = append(args, "--tool-call-parser", req.ToolCallParser)
	}
	if req.EnableAutoToolChoice {
		args = append(args, "--enable-auto-tool-choice")
	}
	if req.ChatTemplate != "" {
		// Convenience: shorthand like "gemma4" resolves to the official
		// template baked into /opt/vllm-templates by the prod image.
		// Anything else is forwarded verbatim so callers can also point
		// at a custom path.
		path := req.ChatTemplate
		if !strings.ContainsAny(path, "/.") {
			path = "/opt/vllm-templates/tool_chat_template_" + path + ".jinja"
		}
		args = append(args, "--chat-template", path)
	}
	if req.EnableThinking {
		args = append(args, "--default-chat-template-kwargs", `{"enable_thinking": true}`)
	}
	return args
}

// resolveModelPath converts a model name to a filesystem path.
func (m *Manager) resolveModelPath(model string) string {
	// If it's already an absolute path, use it.
	if filepath.IsAbs(model) {
		return model
	}
	// Check if it's a directory under modelsDir.
	candidate := filepath.Join(m.modelsDir, model)
	if info, err := os.Stat(candidate); err == nil && info.IsDir() {
		return candidate
	}
	// Otherwise treat as HuggingFace repo name (e.g. "google/gemma-4-31b-it").
	return model
}

// parseProgress reads vLLM stderr and extracts loading progress.
var progressPattern = regexp.MustCompile(`Loading model weights.*?(\d+)%`)
var shardPattern = regexp.MustCompile(`Loading safetensors.*?(\d+)/(\d+)`)

func (m *Manager) parseProgress(r io.Reader, gen uint64) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		// Deliberately NOT re-emitted to our own stdout/stderr: those are
		// the container's log FIFOs, and the 35B floods tens of MB at
		// startup. If the host-side drain ever stalls, the FIFO fills and
		// every subsequent write to it BLOCKS — including the HTTP
		// handlers' log lines, wedging the whole proxy (observed on
		// m2-tdx-gpu, 2026-07-19). vLLM output lives only in the tail
		// ring, surfaced via /v1/models/status.

		// Keep a short tail ring so a process death can report the real
		// error instead of a bare exit code. Gen-guarded: a superseded
		// load's dying output must not pollute the successor's ring —
		// keep DRAINING though, so the dying process never blocks on a
		// full pipe.
		if !m.ifGen(gen, func() {
			m.stderrTail = append(m.stderrTail, line)
			// Keep enough lines to reach past a Python traceback to the actual engine
			// error: vLLM's "Engine core initialization failed. See root cause above."
			// is the tail of a long API-server traceback, and the EngineCore child's
			// real exception is many lines earlier in the same stream.
			if len(m.stderrTail) > 400 {
				m.stderrTail = m.stderrTail[len(m.stderrTail)-400:]
			}
		}) {
			continue
		}

		// Parse percentage progress.
		if matches := progressPattern.FindStringSubmatch(line); len(matches) > 1 {
			var pct float64
			fmt.Sscanf(matches[1], "%f", &pct)
			m.ifGen(gen, func() {
				m.progress = pct / 100.0
				m.message = fmt.Sprintf("Loading weights... %d%%", int(pct))
			})
		}

		// Parse shard progress.
		if matches := shardPattern.FindStringSubmatch(line); len(matches) > 2 {
			var done, total int
			fmt.Sscanf(matches[1], "%d", &done)
			fmt.Sscanf(matches[2], "%d", &total)
			if total > 0 {
				m.ifGen(gen, func() {
					m.progress = float64(done) / float64(total)
					m.message = fmt.Sprintf("Loading weights (%d/%d shards)...", done, total)
				})
			}
		}
	}
}

// waitForReady polls vLLM /health until it returns 200, the process
// dies (waitCh fires → fail fast with a *vllmExitError carrying the
// stderr tail), or the context is cancelled.
func (m *Manager) waitForReady(ctx context.Context, waitCh chan error, gen uint64) error {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf("http://127.0.0.1:%d/health", m.vllmPort)

	m.ifGen(gen, func() { m.message = "Waiting for vLLM to become ready..." })

	// 15-minute cap: vLLM weight load is ~5 min for our largest model
	// but the FIRST inference after a fresh start triggers FlashInfer's
	// per-architecture JIT compile (GDN sm_90a kernels for Qwen Gated
	// DeltaNet, ~66 .cu files via nvcc). On a cold container overlay
	// (e.g. after Spot preempt or `restart`) the JIT cache at
	// /root/.cache/flashinfer is empty and the compile takes ~8 min on
	// top of weight load. Once the cache is warm the same model is
	// ready in ~2 min.
	//
	// We previously waited 30 min here to leave a wide safety margin
	// over a cold compile, but in practice any failure mode worth
	// waiting for already manifests inside ~12 min (cold compile +
	// weight load). Misconfigurations that will never become ready
	// (e.g. max_model_len too large → vLLM OOMs during profile_run)
	// hold the lock the whole time and block other loads / restarts
	// for nothing, so cap at 15 min for the small pooling models.
	//
	// The generate instance gets 25 min: with a fresh container overlay
	// the full cold path stacks FlashInfer JIT (~8 min) + 35 GB weight
	// load off a cold page cache (~6 min) + 262k KV init and CUDA-graph
	// capture (~3-4 min), which lands right ON a 15-min wire (observed
	// twice on m5-dev-ai, 2026-07-15) and burns a whole cycle per miss.
	iterations := 180
	if m.task == TaskGenerate {
		iterations = 300
	}
	for i := 0; i < iterations; i++ {
		select {
		case <-ctx.Done():
			fmt.Fprintf(os.Stderr, "{\"level\":\"warn\",\"msg\":\"waitForReady cancelled\",\"task\":%q,\"iteration\":%d}\n", m.task, i)
			return ctx.Err()
		case waitErr := <-waitCh:
			// The process is gone: give the stderr a beat to drain
			// through parseProgress, then fail with the real reason.
			time.Sleep(500 * time.Millisecond)
			return &vllmExitError{waitErr: waitErr, tail: m.stderrTailSuffix()}
		default:
		}

		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}
		// One diagnostic line a minute: an opaque 15-minute timeout hid a
		// GPU-memory leak loop for hours on m5-dev-ai. Cheap to keep.
		if i%12 == 0 {
			outcome := "no response"
			if err != nil {
				outcome = err.Error()
			} else {
				outcome = fmt.Sprintf("status %d", resp.StatusCode)
			}
			fmt.Fprintf(os.Stderr, "{\"level\":\"info\",\"msg\":\"waitForReady poll\",\"task\":%q,\"url\":%q,\"iteration\":%d,\"outcome\":%q}\n", m.task, url, i, outcome)
		}

		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("vLLM not ready after %d minutes", iterations*5/60)
}

// lookupVerityRoothash returns the dm-verity root hash for the given model
// if disk-mounter has published one to <RoothashDir>/<basename>.roothash.
// The basename is derived from the request model identifier ("gemma4",
// "/models/gemma4", or "google/gemma-4-31b-it" all map to a sensible
// filename). Returns the empty string when no roothash is available, in
// which case the caller falls back to the legacy index-file hash.
func (m *Manager) lookupVerityRoothash(model string) string {
	if m.roothashDir == "" || model == "" {
		return ""
	}
	name := filepath.Base(strings.TrimSuffix(model, "/"))
	if name == "" || name == "." || name == "/" {
		return ""
	}
	data, err := os.ReadFile(filepath.Join(m.roothashDir, name+".roothash"))
	if err != nil {
		return ""
	}
	hash := strings.TrimSpace(string(data))
	// Sanity-check: hex, 40-128 chars (sha1 .. sha512).
	if l := len(hash); l < 40 || l > 128 || l%2 != 0 {
		return ""
	}
	for _, c := range hash {
		if !((c >= '0' && c <= '9') || (c >= 'a' && c <= 'f') || (c >= 'A' && c <= 'F')) {
			return ""
		}
	}
	return hash
}

// computeDigest computes a SHA-256 digest from the model's identity file.
func (m *Manager) computeDigest(modelPath string) string {
	// Resolve HuggingFace cache path if needed.
	resolvedPath := m.resolveHFCachePath(modelPath)

	candidates := []string{
		filepath.Join(resolvedPath, "model.safetensors.index.json"),
		filepath.Join(resolvedPath, ".sha256"),
		filepath.Join(resolvedPath, "config.json"),
	}

	for _, path := range candidates {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		if strings.HasSuffix(path, ".sha256") {
			// Pre-computed digest file.
			return strings.TrimSpace(string(data))
		}
		h := sha256.Sum256(data)
		return hex.EncodeToString(h[:])
	}
	return ""
}

// resolveHFCachePath resolves a HuggingFace model name to its cache snapshot path.
func (m *Manager) resolveHFCachePath(modelPath string) string {
	if info, err := os.Stat(modelPath); err == nil && info.IsDir() {
		return modelPath
	}

	// Try HuggingFace cache structure.
	home, _ := os.UserHomeDir()
	hfHome := os.Getenv("HF_HOME")
	if hfHome == "" {
		hfHome = filepath.Join(home, ".cache", "huggingface")
	}
	repo := "models--" + strings.ReplaceAll(modelPath, "/", "--")
	snapDir := filepath.Join(hfHome, "hub", repo, "snapshots")

	entries, err := os.ReadDir(snapDir)
	if err != nil || len(entries) == 0 {
		return modelPath
	}

	// Return the last (most recent) snapshot.
	return filepath.Join(snapDir, entries[len(entries)-1].Name())
}

// stderrTailSuffix returns a ": <reason>" suffix built from the most
// informative recent stderr lines: prefer explicit error lines (the
// vLLM ValueError / argparse / OOM message), fall back to the last
// non-empty line. Empty when nothing was captured.
func (m *Manager) stderrTailSuffix() string {
	m.mu.RLock()
	tail := make([]string, len(m.stderrTail))
	copy(tail, m.stderrTail)
	m.mu.RUnlock()

	// Scan backwards for the most recent line that looks like the actual
	// error, skipping traceback scaffolding.
	isNoise := func(s string) bool {
		t := strings.TrimSpace(s)
		return t == "" || strings.HasPrefix(t, "File \"") ||
			strings.HasPrefix(t, "Traceback") || strings.HasPrefix(t, "^")
	}
	for i := len(tail) - 1; i >= 0; i-- {
		t := strings.TrimSpace(tail[i])
		if isNoise(t) {
			continue
		}
		lower := strings.ToLower(t)
		if strings.Contains(lower, "error") || strings.Contains(lower, "out of memory") ||
			strings.Contains(lower, "killed") || strings.Contains(lower, "exception") {
			return ": " + t
		}
	}
	for i := len(tail) - 1; i >= 0; i-- {
		if t := strings.TrimSpace(tail[i]); !isNoise(t) {
			return ": " + t
		}
	}
	return ""
}

// setFailed marks the load failed — only when gen is still the current
// load generation, so a superseded load's death never fails its
// successor.
func (m *Manager) setFailed(gen uint64, errMsg string) {
	m.ifGen(gen, func() {
		m.state = StateFailed
		m.loadErr = errMsg
		m.message = "Failed: " + errMsg
	})
}

// persistRequest writes the LoadRequest to stateFile so a future
// RestoreFromDisk call can re-issue it. Best-effort: failures are
// logged-via-message but never returned, since the in-memory load has
// already succeeded.
func (m *Manager) persistRequest(req LoadRequest) {
	if m.stateFile == "" {
		return
	}
	data, err := json.MarshalIndent(req, "", "  ")
	if err != nil {
		return
	}
	tmp := m.stateFile + ".tmp"
	if err := os.MkdirAll(filepath.Dir(m.stateFile), 0o755); err != nil {
		fmt.Fprintf(os.Stderr, "{\"level\":\"warn\",\"msg\":\"persistRequest mkdir failed\",\"error\":%q}\n", err.Error())
		return
	}
	if err := os.WriteFile(tmp, data, 0o600); err != nil {
		fmt.Fprintf(os.Stderr, "{\"level\":\"warn\",\"msg\":\"persistRequest write failed\",\"error\":%q}\n", err.Error())
		return
	}
	if err := os.Rename(tmp, m.stateFile); err != nil {
		fmt.Fprintf(os.Stderr, "{\"level\":\"warn\",\"msg\":\"persistRequest rename failed\",\"error\":%q}\n", err.Error())
	}
}

// RestoreFromDisk reads stateFile (if present) and asynchronously
// re-issues Load. Returns the restored model name, or empty string when
// no state file exists. Errors reading/parsing the file are returned;
// errors from the (background) Load surface via Status().
//
// Call this from main after constructing the Manager. It is safe to
// call before the HTTP server starts: vLLM startup is async, so the
// HTTP server will come up immediately and the model will report
// state=loading until it's ready.
func (m *Manager) RestoreFromDisk() (string, error) {
	req, err := m.readPersistedRequest()
	if err != nil || req == nil {
		return "", err
	}
	if err := m.Load(*req); err != nil {
		return req.Model, fmt.Errorf("auto-load %s: %w", req.Model, err)
	}
	return req.Model, nil
}

// readPersistedRequest reads and validates the persisted LoadRequest
// without issuing a Load. Returns (nil, nil) when persistence is
// disabled or no state file exists.
func (m *Manager) readPersistedRequest() (*LoadRequest, error) {
	if m.stateFile == "" {
		return nil, nil
	}
	data, err := os.ReadFile(m.stateFile)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, fmt.Errorf("read %s: %w", m.stateFile, err)
	}
	var req LoadRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return nil, fmt.Errorf("parse %s: %w", m.stateFile, err)
	}
	if req.Model == "" {
		return nil, fmt.Errorf("state file %s has empty model", m.stateFile)
	}
	return &req, nil
}
