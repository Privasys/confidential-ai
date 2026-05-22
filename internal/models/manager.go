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
}

// Status is the response for GET /v1/models/status.
type Status struct {
	State       State   `json:"state"`
	Model       string  `json:"model,omitempty"`
	ModelDigest string  `json:"model_digest,omitempty"`
	Progress    float64 `json:"progress,omitempty"`    // 0.0 - 1.0 during loading
	Message     string  `json:"message,omitempty"`     // human-readable status
	ElapsedSec  float64 `json:"elapsed_s,omitempty"`
	Error       string  `json:"error,omitempty"`
}

// Manager manages the vLLM subprocess lifecycle.
type Manager struct {
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
}

// NewManager creates a new model manager. stateFile may be empty to
// disable persistence (in-process testing). When non-empty the file is
// written after every successful Load and read by RestoreFromDisk.
func NewManager(modelsDir string, vllmPort int, roothashDir, stateFile string) *Manager {
	return &Manager{
		modelsDir:   modelsDir,
		vllmPort:    vllmPort,
		roothashDir: roothashDir,
		stateFile:   stateFile,
		state:       StateIdle,
	}
}

// Status returns the current model manager status.
func (m *Manager) Status() Status {
	m.mu.RLock()
	defer m.mu.RUnlock()

	s := Status{
		State:       m.state,
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

// Load starts loading a model. Returns an error if already loading.
// Idempotent: if the requested model is already loaded, returns nil.
func (m *Manager) Load(req LoadRequest) error {
	m.mu.Lock()

	// Idempotent: already loaded same model.
	if m.state == StateReady && m.model == req.Model {
		m.mu.Unlock()
		return nil
	}

	// Already loading.
	if m.state == StateLoading {
		m.mu.Unlock()
		return fmt.Errorf("already loading model %q", m.model)
	}

	// If a different model is loaded, unload first.
	if m.state == StateReady && m.model != req.Model {
		m.mu.Unlock()
		if err := m.Unload(); err != nil {
			return fmt.Errorf("unload current model: %w", err)
		}
		m.mu.Lock()
	}

	// Apply defaults.
	if req.Dtype == "" {
		req.Dtype = "auto"
	}
	if req.MaxModelLen == 0 {
		req.MaxModelLen = 8192
	}
	if req.GPUMemoryUtilization == 0 {
		req.GPUMemoryUtilization = 0.90
	}

	m.state = StateLoading
	m.model = req.Model
	m.quantization = req.Quantization
	m.modelDigest = ""
	m.progress = 0
	m.message = "Starting vLLM..."
	m.loadStart = time.Now()
	m.loadErr = ""
	m.mu.Unlock()

	go m.doLoad(req)
	return nil
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
	cancel := m.cancel
	cmd := m.cmd
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
		// cmd.WaitDelay (set in doLoad) bounds this at 10 s even if a
		// grandchild keeps the pipe open.
		_ = cmd.Wait()
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

// doLoad runs the vLLM subprocess and tracks its progress.
func (m *Manager) doLoad(req LoadRequest) {
	ctx, cancel := context.WithCancel(context.Background())
	m.mu.Lock()
	m.cancel = cancel
	m.mu.Unlock()

	defer func() {
		cancel()
		m.mu.Lock()
		m.cancel = nil
		m.mu.Unlock()
	}()

	// Resolve model path. Source overrides Model when present, so the
	// served name (the canonical id reported back to clients) can be
	// short and friendly while the loader still finds the on-disk
	// safetensors directory.
	loaderID := req.Source
	if loaderID == "" {
		loaderID = req.Model
	}
	modelPath := m.resolveModelPath(loaderID)

	// Build vLLM command.
	//
	// Reproducibility tradeoffs we accept:
	//
	//   * CUDA graphs ENABLED (no --enforce-eager). Graph capture +
	//     replay is itself deterministic: same kernel DAG, same
	//     memory offsets, same streams. Eager mode was costing ~30x
	//     throughput for no determinism gain in our locked
	//     environment (pinned kernel, NVIDIA driver, CUDA, vLLM
	//     version, model digest).
	//
	//   * V1 engine ENABLED (no VLLM_USE_V1=0 env). The legacy V0
	//     scheduler underutilises the H100 tensor cores; V1 was
	//     historically avoided for reproducibility because of
	//     chunked prefill (which can change the order of FP
	//     reductions across prefill chunks), so we explicitly turn
	//     chunked prefill off below and size the batch budget so
	//     any prompt up to --max-model-len fits in a single
	//     mathematical block.
	//
	//   * --max-num-batched-tokens is set to max(16384, 2*max-model-len)
	//     so the entire prompt is processed in one prefill step
	//     (no inter-chunk reductions).
	//
	// What this still does NOT guarantee: batch-invariance. With
	// continuous batching, two concurrent requests may see
	// different reductions across the batch dimension. Determinism
	// holds per-request when traffic is serialised; for true
	// concurrent determinism we need batch-invariant kernels
	// (FlashInfer / vLLM batch-invariant attention), tracked
	// separately.
	batchedTokens := req.MaxModelLen * 2
	if batchedTokens < 16384 {
		batchedTokens = 16384
	}

	// Sensible defaults for known model families. Callers (the seed
	// scripts, the management-service, ad-hoc /v1/models/load) don't
	// have to know which architecture-specific reasoning + tool
	// parsers vLLM ships; if the canonical id contains "gemma4" we
	// auto-wire the official Gemma 4 thinking recipe (see
	// https://docs.vllm.ai/projects/recipes/en/latest/Google/Gemma4.html).
	// Explicit fields on LoadRequest still win, so anyone can override
	// (e.g. to disable thinking-by-default or point at a custom
	// chat template).
	if strings.Contains(strings.ToLower(req.Model), "gemma4") {
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

	// Qwen3 / Qwen3.5 / Qwen3.6 are thinking models: without
	// `--reasoning-parser qwen3` vLLM emits the entire `<think>…</think>`
	// block as plain `content`, the chat UI's splitReasoning() routes it
	// to the Thought-process panel, and any reply where the model puts
	// everything inside <think> (a common Qwen3 failure mode for short
	// agentic prompts) renders as an empty assistant message. Tool calls
	// also need `--tool-call-parser hermes` (Qwen3 reuses the Hermes
	// tool-call schema) plus `--enable-auto-tool-choice`. The same
	// override-wins-if-set policy as gemma4 above.
	lname := strings.ToLower(req.Model)
	if strings.Contains(lname, "qwen3") || strings.Contains(lname, "qwen36") || strings.Contains(lname, "qwen35") {
		if req.ReasoningParser == "" {
			req.ReasoningParser = "qwen3"
		}
		if req.ToolCallParser == "" {
			req.ToolCallParser = "hermes"
			req.EnableAutoToolChoice = true
		}
		if !req.EnableThinking {
			req.EnableThinking = true
		}
	}

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
		"--no-enable-chunked-prefill",
		"--max-num-batched-tokens", fmt.Sprintf("%d", batchedTokens),
		"--no-enable-log-requests",
		"--max-model-len", fmt.Sprintf("%d", req.MaxModelLen),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", req.GPUMemoryUtilization),
		"--dtype", req.Dtype,
		"--port", fmt.Sprintf("%d", m.vllmPort),
	}
	if req.MaxNumSeqs > 0 {
		args = append(args, "--max-num-seqs", fmt.Sprintf("%d", req.MaxNumSeqs))
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

	cmd := exec.CommandContext(ctx, "vllm", args...)
	cmd.Env = append(os.Environ(),
		// Force PyTorch / cuBLAS into deterministic-workspace mode.
		// Required for torch.use_deterministic_algorithms() under
		// CUDA >= 10.2; vLLM picks this up via PyTorch.
		"CUBLAS_WORKSPACE_CONFIG=:4096:8",
		"PYTHONHASHSEED=0",
		// VLLM_USE_V1 intentionally NOT set: defaults to V1 in
		// vLLM >= 0.19, which is what we want for throughput.
	)
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
		m.setFailed("failed to create stderr pipe: " + err.Error())
		return
	}
	cmd.Stdout = os.Stdout // let vLLM stdout pass through

	m.mu.Lock()
	m.cmd = cmd
	m.message = "Starting vLLM process..."
	m.mu.Unlock()

	if err := cmd.Start(); err != nil {
		m.setFailed("failed to start vLLM: " + err.Error())
		return
	}

	// Parse stderr for progress in background.
	go m.parseProgress(stderr)

	// Wait for vLLM to be ready.
	if err := m.waitForReady(ctx); err != nil {
		// Check if process is still alive.
		if cmd.ProcessState != nil && cmd.ProcessState.Exited() {
			m.setFailed("vLLM exited during startup: exit code " + fmt.Sprintf("%d", cmd.ProcessState.ExitCode()))
		} else {
			m.setFailed("vLLM health check failed: " + err.Error())
			_ = cmd.Process.Kill()
			_ = cmd.Wait()
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

	m.mu.Lock()
	m.state = StateReady
	m.model = req.Model
	m.modelDigest = digest
	m.progress = 1.0
	m.message = "Model loaded and serving"
	m.mu.Unlock()

	// Persist the successful load so a container restart auto-recovers.
	m.persistRequest(req)

	// Wait for process exit (blocks until vLLM dies or is killed).
	if err := cmd.Wait(); err != nil {
		m.mu.Lock()
		if m.state == StateReady {
			// Unexpected crash.
			m.state = StateFailed
			m.loadErr = "vLLM process exited unexpectedly: " + err.Error()
		}
		m.mu.Unlock()
	}
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

func (m *Manager) parseProgress(r io.Reader) {
	scanner := bufio.NewScanner(r)
	for scanner.Scan() {
		line := scanner.Text()
		// Write to our stderr for logging.
		fmt.Fprintln(os.Stderr, "[vllm]", line)

		// Parse percentage progress.
		if matches := progressPattern.FindStringSubmatch(line); len(matches) > 1 {
			var pct float64
			fmt.Sscanf(matches[1], "%f", &pct)
			m.mu.Lock()
			m.progress = pct / 100.0
			m.message = fmt.Sprintf("Loading weights... %d%%", int(pct))
			m.mu.Unlock()
		}

		// Parse shard progress.
		if matches := shardPattern.FindStringSubmatch(line); len(matches) > 2 {
			var done, total int
			fmt.Sscanf(matches[1], "%d", &done)
			fmt.Sscanf(matches[2], "%d", &total)
			if total > 0 {
				m.mu.Lock()
				m.progress = float64(done) / float64(total)
				m.message = fmt.Sprintf("Loading weights (%d/%d shards)...", done, total)
				m.mu.Unlock()
			}
		}
	}
}

// waitForReady polls vLLM /health until it returns 200 or context is cancelled.
func (m *Manager) waitForReady(ctx context.Context) error {
	client := &http.Client{Timeout: 5 * time.Second}
	url := fmt.Sprintf("http://localhost:%d/health", m.vllmPort)

	m.mu.Lock()
	m.message = "Waiting for vLLM to become ready..."
	m.mu.Unlock()

	// 30-minute cap: vLLM weight load is ~5 min for our largest model
	// but the FIRST inference after a fresh start triggers FlashInfer's
	// per-architecture JIT compile (GDN sm_90a kernels for Qwen Gated
	// DeltaNet, ~66 .cu files via nvcc). On a cold container overlay
	// (e.g. after Spot preempt or `restart`) the JIT cache at
	// /root/.cache/flashinfer is empty and the compile takes ~8 min on
	// top of weight load — well past the old 10 min cap. Once the cache
	// is warm the same model is ready in ~2 min.
	for i := 0; i < 360; i++ {
		select {
		case <-ctx.Done():
			return ctx.Err()
		default:
		}

		resp, err := client.Get(url)
		if err == nil {
			resp.Body.Close()
			if resp.StatusCode == http.StatusOK {
				return nil
			}
		}

		time.Sleep(5 * time.Second)
	}
	return fmt.Errorf("vLLM not ready after 30 minutes")
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

func (m *Manager) setFailed(errMsg string) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.state = StateFailed
	m.loadErr = errMsg
	m.message = "Failed: " + errMsg
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
	if m.stateFile == "" {
		return "", nil
	}
	data, err := os.ReadFile(m.stateFile)
	if err != nil {
		if os.IsNotExist(err) {
			return "", nil
		}
		return "", fmt.Errorf("read %s: %w", m.stateFile, err)
	}
	var req LoadRequest
	if err := json.Unmarshal(data, &req); err != nil {
		return "", fmt.Errorf("parse %s: %w", m.stateFile, err)
	}
	if req.Model == "" {
		return "", fmt.Errorf("state file %s has empty model", m.stateFile)
	}
	if err := m.Load(req); err != nil {
		return req.Model, fmt.Errorf("auto-load %s: %w", req.Model, err)
	}
	return req.Model, nil
}
