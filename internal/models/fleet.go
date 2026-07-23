package models

import (
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"
)

// Fleet coordinates up to three vLLM instances on one GPU: the chat LLM
// (generate) plus the small pooling models Drive's in-enclave RAG needs
// (embed + rerank). Each task is its own Manager / subprocess / port:
//
//	generate  basePort     (historically 8000)
//	embed     basePort + 1
//	rerank    basePort + 2
//
// Loads are SEQUENCED, main model first: vLLM sizes its memory pool from
// a startup profile run, so the ~14-min FlashInfer cold start of the main
// LLM must claim its 0.86 utilisation slice before the small instances
// grab theirs (~0.05 each, seconds to load). The configure-then-freeze
// gate covers ALL configured models: IsReady() is true only when every
// desired instance is serving, and Status() reports WHICH model is
// currently configuring/loading so the portal/CLI can follow multi-model
// startup (app-status Phase 1).
type Fleet struct {
	modelsDir string

	// instances is fixed at construction; Managers are never replaced,
	// only loaded/unloaded, so per-instance methods need no lock.
	instances map[Task]*Manager

	mu sync.Mutex
	// desired is the declaratively-configured task set: tasks with a
	// request applied by Apply. Readiness quantifies over it.
	desired map[Task]LoadRequest
	// seq invalidates an in-flight load sequence when a newer Apply
	// supersedes it.
	seq int
	// queued marks tasks whose load is accepted but not yet issued
	// (waiting for an earlier instance in the order). Status surfaces
	// them as loading with an explanatory message.
	queued map[Task]string
	// seqDone closes when the most-recently-spawned sequence goroutine
	// exits. Each new sequence WAITS on its predecessor before touching
	// any Manager, so two sequences can never interleave loads — the
	// 2026-07-23 dev incident had a boot-restore sequence and an owner
	// configure racing, which let a pooling engine profile its memory
	// while the chat engine was mid-warmup (KV check read -9.02 GiB).
	seqDone chan struct{}
}

// taskOrder is the load sequence: main model first (see VRAM plan).
var taskOrder = []Task{TaskGenerate, TaskEmbed, TaskRerank}

// NewFleet constructs the three per-task Managers. stateFile is the
// generate instance's persistence path (back-compat: /data/last-load.json);
// the pooling instances persist next to it as last-load-<task>.json.
// Empty stateFile disables persistence for all instances.
func NewFleet(modelsDir string, basePort int, roothashDir, stateFile string) *Fleet {
	stateFor := func(task Task) string {
		if stateFile == "" || task == TaskGenerate {
			return stateFile
		}
		dir, base := filepath.Split(stateFile)
		ext := filepath.Ext(base)
		return filepath.Join(dir, strings.TrimSuffix(base, ext)+"-"+string(task)+ext)
	}
	f := &Fleet{
		modelsDir: modelsDir,
		instances: map[Task]*Manager{},
		desired:   map[Task]LoadRequest{},
		queued:    map[Task]string{},
	}
	for i, task := range taskOrder {
		f.instances[task] = NewManager(task, modelsDir, basePort+i, roothashDir, stateFor(task))
	}
	return f
}

// Instance returns the Manager for a task (always non-nil for known tasks).
func (f *Fleet) Instance(task Task) *Manager { return f.instances[task] }

// Generate is shorthand for the chat-LLM instance, whose identity the
// legacy single-model surfaces (top-level status fields, OID 3.5,
// reproducibility block) keep reporting.
func (f *Fleet) Generate() *Manager { return f.instances[TaskGenerate] }

// ListAvailable returns model directories found under modelsDir.
func (f *Fleet) ListAvailable() ([]string, error) {
	entries, err := os.ReadDir(f.modelsDir)
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

// Apply reconciles the fleet towards the requested task set.
//
// declarative=true (the configure path) treats reqs as the COMPLETE
// desired state: tasks absent from the map are unloaded. declarative=
// false (the ad-hoc single-task path) touches only the listed tasks.
//
// Loads are issued asynchronously in taskOrder, each waiting for the
// previous instance to settle (ready/failed) before starting, so the
// main model claims its memory slice first. A later Apply supersedes an
// in-flight sequence at the next step boundary.
func (f *Fleet) Apply(reqs map[Task]LoadRequest, declarative bool) error {
	// Validate before mutating anything.
	for task, req := range reqs {
		norm, err := NormalizeTask(task)
		if err != nil {
			return err
		}
		if norm != task {
			return fmt.Errorf("internal: task key %q not normalized", task)
		}
		if req.Model == "" {
			return fmt.Errorf("task %s: model field is required", task)
		}
	}

	f.mu.Lock()
	f.seq++
	seq := f.seq
	var unloads []Task
	for _, task := range taskOrder {
		req, want := reqs[task]
		if want {
			f.desired[task] = req
			f.queued[task] = "Queued: waiting for earlier models to load"
		} else if declarative {
			if _, had := f.desired[task]; had {
				unloads = append(unloads, task)
			}
			delete(f.desired, task)
			delete(f.queued, task)
		}
	}
	// Single-flight: chain onto the previous sequence goroutine. The
	// predecessor aborts at its next step boundary (superseded via seq),
	// and this sequence starts only once it has fully exited, so Manager
	// calls from two sequences never interleave.
	prev := f.seqDone
	done := make(chan struct{})
	f.seqDone = done
	f.mu.Unlock()

	go func() {
		defer close(done)
		if prev != nil {
			<-prev
		}
		f.runSequence(seq, reqs, unloads)
	}()
	return nil
}

// runSequence executes one Apply's unloads + ordered loads. It aborts at
// a step boundary when a newer Apply has bumped seq.
func (f *Fleet) runSequence(seq int, reqs map[Task]LoadRequest, unloads []Task) {
	superseded := func() bool {
		f.mu.Lock()
		defer f.mu.Unlock()
		return f.seq != seq
	}

	// Unloads first: they free VRAM the loads are about to claim.
	for _, task := range unloads {
		if superseded() {
			return
		}
		_ = f.instances[task].Unload()
	}

	for _, task := range taskOrder {
		req, want := reqs[task]
		if !want {
			continue
		}
		if superseded() {
			return
		}
		inst := f.instances[task]

		// Wait out any in-flight load on this instance. Manager.Load can
		// supersede a running load, but letting it settle first keeps the
		// GPU accounting clean: an obsolete load's memory claim must be
		// released (or become steady-state) before the next engine
		// profiles its slice.
		if !f.awaitSettled(inst, seq) {
			return
		}
		f.mu.Lock()
		delete(f.queued, task)
		f.mu.Unlock()
		if err := inst.Load(req); err != nil {
			// Surfaced via the instance's Status on the next poll; the
			// sequence continues so one bad instance doesn't wedge the
			// rest (readiness still fails closed on the desired set).
			continue
		}
		// Sequencing invariant: the next instance starts only after this
		// one settles. Ready → proceed; failed → proceed too (readiness
		// already fails closed; loading the small models is still useful
		// for diagnosis and a retried configure gets idempotent no-ops).
		if !f.awaitSettled(inst, seq) {
			return
		}
	}
}

// awaitSettled blocks until inst is not loading, the sequence is
// superseded (returns false), or a 20-minute safety cap elapses (the
// Manager itself caps a load at 15 min).
func (f *Fleet) awaitSettled(inst *Manager, seq int) bool {
	deadline := time.Now().Add(20 * time.Minute)
	for {
		f.mu.Lock()
		live := f.seq == seq
		f.mu.Unlock()
		if !live {
			return false
		}
		if inst.Status().State != StateLoading {
			return true
		}
		if time.Now().After(deadline) {
			return true // let the caller try; Manager will refuse if still loading
		}
		time.Sleep(2 * time.Second)
	}
}

// UnloadAll stops every instance and clears the desired set.
func (f *Fleet) UnloadAll() error {
	f.mu.Lock()
	f.seq++
	f.desired = map[Task]LoadRequest{}
	f.queued = map[Task]string{}
	f.mu.Unlock()
	var firstErr error
	for _, task := range taskOrder {
		if err := f.instances[task].Unload(); err != nil && firstErr == nil {
			firstErr = err
		}
	}
	return firstErr
}

// Unload stops a single instance and removes it from the desired set.
func (f *Fleet) Unload(task Task) error {
	task, err := NormalizeTask(task)
	if err != nil {
		return err
	}
	f.mu.Lock()
	f.seq++
	delete(f.desired, task)
	delete(f.queued, task)
	f.mu.Unlock()
	return f.instances[task].Unload()
}

// FleetStatus is the aggregate response for GET /v1/models/status. The
// embedded Status keeps the legacy single-model wire shape (state /
// model / model_digest / progress / message) that the enclave-os
// runtime-status poller folds into the app-status protocol: State is the
// AGGREGATE (ready only when every desired instance is ready — the
// freeze covers all models), while Model/ModelDigest identify the
// generate instance. Message names WHICH model is currently loading.
// Models carries the full per-task detail.
type FleetStatus struct {
	Status
	Models map[Task]Status `json:"models,omitempty"`
}

// loadWeight approximates each task's share of total startup time for
// the aggregate progress bar: the main LLM dominates (cold FlashInfer
// JIT ~14 min) while the 0.6B pooling models load in seconds.
var loadWeight = map[Task]float64{TaskGenerate: 0.9, TaskEmbed: 0.05, TaskRerank: 0.05}

// Status returns the aggregate + per-task status document.
func (f *Fleet) Status() FleetStatus {
	f.mu.Lock()
	desired := make(map[Task]LoadRequest, len(f.desired))
	for k, v := range f.desired {
		desired[k] = v
	}
	queued := make(map[Task]string, len(f.queued))
	for k, v := range f.queued {
		queued[k] = v
	}
	f.mu.Unlock()

	per := map[Task]Status{}
	for _, task := range taskOrder {
		s := f.instances[task].Status()
		if msg, isQueued := queued[task]; isQueued && (s.State == StateIdle || s.State == StateFailed) {
			// Accepted but not yet issued (an earlier instance is still
			// loading). Present as loading so pollers see one continuous
			// startup, with an explanatory message. A FAILED instance that
			// is queued is a retry in flight: presenting the stale failure
			// made a re-delivered configure look like it did nothing (the
			// orchestrator snapshotted "failed" while the retry was still
			// queued behind the main model — 2026-07-23 dev incident).
			if s.State == StateFailed {
				msg = "Queued: retrying after failure"
			}
			s.State = StateLoading
			s.Model = desired[task].Model
			s.Message = msg
			s.Error = ""
			s.StderrTail = nil
		}
		per[task] = s
	}

	agg := FleetStatus{Models: per}

	// Legacy identity fields: the generate instance.
	gen := per[TaskGenerate]
	agg.Task = TaskGenerate
	agg.Model = gen.Model
	agg.ModelDigest = gen.ModelDigest

	if len(desired) == 0 {
		// Nothing configured: mirror the generate instance verbatim so
		// ad-hoc single-instance use (dev) behaves exactly as before.
		agg.State = gen.State
		agg.Progress = gen.Progress
		agg.Message = gen.Message
		agg.ElapsedSec = gen.ElapsedSec
		agg.Error = gen.Error
		return agg
	}

	// Aggregate over the desired set: failed > loading > ready.
	state := StateReady
	var progress, totalWeight float64
	var messages []string
	var errs []string
	for _, task := range taskOrder {
		if _, want := desired[task]; !want {
			continue
		}
		s := per[task]
		w := loadWeight[task]
		totalWeight += w
		switch s.State {
		case StateReady:
			progress += w
		case StateLoading:
			if state != StateFailed {
				state = StateLoading
			}
			progress += w * s.Progress
			if s.Message != "" {
				messages = append(messages, s.Model+": "+s.Message)
			}
			if s.ElapsedSec > agg.ElapsedSec {
				agg.ElapsedSec = s.ElapsedSec
			}
		case StateFailed:
			state = StateFailed
			errs = append(errs, s.Model+": "+s.Error)
		case StateIdle:
			// Desired but not started (e.g. sequence aborted): count as
			// in-progress at zero.
			if state != StateFailed {
				state = StateLoading
			}
			if s.Model == "" {
				messages = append(messages, desired[task].Model+": waiting to load")
			}
		}
	}

	agg.State = state
	if totalWeight > 0 {
		agg.Progress = progress / totalWeight
	}
	switch state {
	case StateReady:
		agg.Progress = 1.0
		agg.Message = "All models loaded and serving"
	case StateLoading:
		agg.Message = strings.Join(messages, "; ")
	case StateFailed:
		agg.Error = strings.Join(errs, "; ")
		agg.Message = "Failed: " + agg.Error
	}
	return agg
}

// IsReady reports whether every desired instance is serving. With an
// empty desired set (nothing configured yet) it falls back to the
// generate instance so ad-hoc dev flows keep working.
func (f *Fleet) IsReady() bool {
	f.mu.Lock()
	desired := make([]Task, 0, len(f.desired))
	for task := range f.desired {
		desired = append(desired, task)
	}
	f.mu.Unlock()
	if len(desired) == 0 {
		return f.instances[TaskGenerate].IsReady()
	}
	for _, task := range desired {
		if !f.instances[task].IsReady() {
			return false
		}
	}
	return true
}

// TaskReady reports whether one task's instance is serving.
func (f *Fleet) TaskReady(task Task) bool {
	m, ok := f.instances[task]
	return ok && m.IsReady()
}

// RestoreFromDisk re-issues the last successful load of every instance
// whose state file survives on the encrypted volume, sequenced like any
// other Apply. Returns the restored model names keyed by task.
func (f *Fleet) RestoreFromDisk() (map[Task]string, error) {
	reqs := map[Task]LoadRequest{}
	restored := map[Task]string{}
	var firstErr error
	for _, task := range taskOrder {
		req, err := f.instances[task].readPersistedRequest()
		if err != nil {
			if firstErr == nil {
				firstErr = fmt.Errorf("%s: %w", task, err)
			}
			continue
		}
		if req == nil {
			continue
		}
		req.Task = task
		reqs[task] = *req
		restored[task] = req.Model
	}
	if len(reqs) == 0 {
		return restored, firstErr
	}
	if err := f.Apply(reqs, false); err != nil && firstErr == nil {
		firstErr = err
	}
	return restored, firstErr
}
