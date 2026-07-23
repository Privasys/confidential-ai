package models

import (
	"os"
	"path/filepath"
	"testing"
	"time"
)

// TestNormalizeIdempotent pins the persist/compare symmetry that broke on
// 2026-07-23: the effective request must be a fixed point of
// normalizeRequest, so a persisted (already-normalized) request restored
// from disk compares EQUAL to a re-delivered identical wire request.
// When recipes were applied after appliedReq was stored, every
// restore→re-configure compared unequal and spuriously reloaded a
// serving model.
func TestNormalizeIdempotent(t *testing.T) {
	cases := []struct {
		task Task
		req  LoadRequest
	}{
		{TaskGenerate, LoadRequest{Model: "qwen36-35b-a3b-fp8", MaxModelLen: 262144, GPUMemoryUtilization: 0.82, KVCacheDtype: "fp8", EnableChunkedPrefill: true, MaxNumBatchedTokens: 32768, MaxNumSeqs: 4, EnablePrefixCaching: true}},
		{TaskGenerate, LoadRequest{Model: "gemma4-27b"}},
		{TaskEmbed, LoadRequest{Model: "qwen3-embedding-06b"}},
		{TaskRerank, LoadRequest{Model: "qwen3-reranker-06b"}},
	}
	for _, c := range cases {
		m := NewManager(c.task, t.TempDir(), 18000, "", "")
		once := m.normalizeRequest(c.req)
		twice := m.normalizeRequest(once)
		if once != twice {
			t.Errorf("%s/%s: normalize not idempotent:\n once=%+v\ntwice=%+v", c.task, c.req.Model, once, twice)
		}
	}
}

// TestNormalizeAppliesRecipes verifies the family recipes moved from
// doLoad into normalizeRequest (i.e. BEFORE the idempotency compare).
func TestNormalizeAppliesRecipes(t *testing.T) {
	g := NewManager(TaskGenerate, t.TempDir(), 18000, "", "")
	q := g.normalizeRequest(LoadRequest{Model: "qwen36-35b-a3b-fp8"})
	if q.ReasoningParser != "qwen3" || q.ToolCallParser != "qwen3_coder" || !q.EnableAutoToolChoice {
		t.Errorf("qwen recipe not applied in normalize: %+v", q)
	}
	r := NewManager(TaskRerank, t.TempDir(), 18002, "", "")
	rr := r.normalizeRequest(LoadRequest{Model: "qwen3-reranker-06b"})
	if rr.HFOverrides == "" {
		t.Errorf("reranker hf_overrides not applied in normalize: %+v", rr)
	}
	// Pooling instances must NOT pick up the chat recipes.
	e := NewManager(TaskEmbed, t.TempDir(), 18001, "", "")
	ee := e.normalizeRequest(LoadRequest{Model: "qwen3-embedding-06b"})
	if ee.ReasoningParser != "" || ee.ToolCallParser != "" {
		t.Errorf("embed instance picked up chat recipes: %+v", ee)
	}
}

// fakeVLLM puts a fake `vllm` binary (sleeps forever) on PATH so Load's
// subprocess spawn succeeds and the manager sits in StateLoading.
func fakeVLLM(t *testing.T) {
	t.Helper()
	dir := t.TempDir()
	script := filepath.Join(dir, "vllm")
	if err := os.WriteFile(script, []byte("#!/bin/sh\nsleep 600\n"), 0o755); err != nil {
		t.Fatal(err)
	}
	t.Setenv("PATH", dir+string(os.PathListSeparator)+os.Getenv("PATH"))
}

func waitState(t *testing.T, m *Manager, want State, d time.Duration) {
	t.Helper()
	deadline := time.Now().Add(d)
	for time.Now().Before(deadline) {
		if m.Status().State == want {
			return
		}
		time.Sleep(20 * time.Millisecond)
	}
	t.Fatalf("state never became %s (now %s: %+v)", want, m.Status().State, m.Status())
}

// TestLoadJoinsIdenticalInFlight: re-delivering the identical request
// while a load is in flight must JOIN it (nil, no restart), not error
// with "already loading" — a re-issued configure during startup used to
// surface that error as a spurious failure.
func TestLoadJoinsIdenticalInFlight(t *testing.T) {
	fakeVLLM(t)
	m := NewManager(TaskEmbed, t.TempDir(), 18101, "", "")
	req := LoadRequest{Model: "m1"}
	if err := m.Load(req); err != nil {
		t.Fatalf("first load: %v", err)
	}
	waitState(t, m, StateLoading, 2*time.Second)
	if err := m.Load(req); err != nil {
		t.Fatalf("identical re-load must join, got error: %v", err)
	}
	if s := m.Status(); s.State != StateLoading || s.Model != "m1" {
		t.Fatalf("join changed state: %+v", s)
	}
	_ = m.Unload()
}

// TestLoadSupersedesDifferentInFlight: a different request supersedes the
// in-flight load, and the DYING load's exit must not clobber the
// successor (the generation guard). Before the guard, the superseded
// process's death marked the replacement load failed — the 2026-07-23
// dev incident.
func TestLoadSupersedesDifferentInFlight(t *testing.T) {
	fakeVLLM(t)
	m := NewManager(TaskEmbed, t.TempDir(), 18102, "", "")
	if err := m.Load(LoadRequest{Model: "m1"}); err != nil {
		t.Fatalf("first load: %v", err)
	}
	waitState(t, m, StateLoading, 2*time.Second)

	if err := m.Load(LoadRequest{Model: "m2"}); err != nil {
		t.Fatalf("superseding load: %v", err)
	}
	// The successor owns the state; give the dying first load ample time
	// to be reaped and (incorrectly) write state if unguarded.
	time.Sleep(1500 * time.Millisecond)
	s := m.Status()
	if s.State != StateLoading || s.Model != "m2" {
		t.Fatalf("superseded load clobbered successor: %+v", s)
	}
	_ = m.Unload()
}

// TestUnloadSilencesInFlightLoad: Unload during a load must leave the
// instance Idle; the killed load's failure path must not resurrect a
// Failed state afterwards.
func TestUnloadSilencesInFlightLoad(t *testing.T) {
	fakeVLLM(t)
	m := NewManager(TaskEmbed, t.TempDir(), 18103, "", "")
	if err := m.Load(LoadRequest{Model: "m1"}); err != nil {
		t.Fatalf("load: %v", err)
	}
	waitState(t, m, StateLoading, 2*time.Second)
	if err := m.Unload(); err != nil {
		t.Fatalf("unload: %v", err)
	}
	time.Sleep(1500 * time.Millisecond)
	if s := m.Status(); s.State != StateIdle {
		t.Fatalf("killed load wrote state after Unload: %+v", s)
	}
}
