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
	"fmt"
	"io"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"strings"
	"sync"
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
type LoadRequest struct {
	Model                 string  `json:"model"`                            // directory name under /models, or HF repo
	Dtype                 string  `json:"dtype,omitempty"`                  // default "auto"
	Quantization          string  `json:"quantization,omitempty"`           // awq, gptq, int4, etc.
	MaxModelLen           int     `json:"max_model_len,omitempty"`          // default 8192
	GPUMemoryUtilization  float64 `json:"gpu_memory_utilization,omitempty"` // default 0.90
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
	modelsDir string // path to model directory (e.g. /models)
	vllmPort  int    // port for vLLM to listen on

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

// NewManager creates a new model manager.
func NewManager(modelsDir string, vllmPort int) *Manager {
	return &Manager{
		modelsDir: modelsDir,
		vllmPort:  vllmPort,
		state:     StateIdle,
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
func (m *Manager) Unload() error {
	m.mu.Lock()
	defer m.mu.Unlock()

	if m.state == StateIdle {
		return nil
	}

	if m.cancel != nil {
		m.cancel()
		m.cancel = nil
	}
	if m.cmd != nil && m.cmd.Process != nil {
		_ = m.cmd.Process.Kill()
		_ = m.cmd.Wait()
	}

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

	// Resolve model path.
	modelPath := m.resolveModelPath(req.Model)

	// Build vLLM command.
	args := []string{
		"serve", modelPath,
		"--seed", "0",
		"--tensor-parallel-size", "1",
		"--enforce-eager",
		"--no-enable-log-requests",
		"--max-model-len", fmt.Sprintf("%d", req.MaxModelLen),
		"--gpu-memory-utilization", fmt.Sprintf("%.2f", req.GPUMemoryUtilization),
		"--dtype", req.Dtype,
		"--port", fmt.Sprintf("%d", m.vllmPort),
	}
	if req.Quantization != "" && req.Quantization != "none" {
		args = append(args, "--quantization", req.Quantization)
	}

	cmd := exec.CommandContext(ctx, "vllm", args...)
	cmd.Env = append(os.Environ(),
		"VLLM_USE_V1=0",
		"CUBLAS_WORKSPACE_CONFIG=:4096:8",
		"PYTHONHASHSEED=0",
	)

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

	// Compute model digest.
	digest := m.computeDigest(modelPath)

	// Derive display name.
	displayName := req.Model
	if strings.HasPrefix(displayName, "/models/") {
		displayName = strings.TrimPrefix(displayName, "/models/")
	}

	m.mu.Lock()
	m.state = StateReady
	m.model = displayName
	m.modelDigest = digest
	m.progress = 1.0
	m.message = "Model loaded and serving"
	m.mu.Unlock()

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

	for i := 0; i < 120; i++ {
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
	return fmt.Errorf("vLLM not ready after 10 minutes")
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
