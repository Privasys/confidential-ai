package models

import (
	"encoding/json"
	"testing"
)

func TestLoadRequest_UnmarshalStringOrNumber(t *testing.T) {
	// The portal Configure form submits every field as a string.
	var r LoadRequest
	if err := json.Unmarshal([]byte(`{"model":"m","dtype":"auto","quantization":"","max_model_len":"262144","gpu_memory_utilization":"0.93","max_num_seqs":"4"}`), &r); err != nil {
		t.Fatalf("string numerics: %v", err)
	}
	if r.Model != "m" || r.Dtype != "auto" {
		t.Errorf("string fields lost: %+v", r)
	}
	if r.MaxModelLen != 262144 || r.MaxNumSeqs != 4 || r.GPUMemoryUtilization != 0.93 {
		t.Errorf("numeric strings not parsed: len=%d seqs=%d gpu=%v", r.MaxModelLen, r.MaxNumSeqs, r.GPUMemoryUtilization)
	}

	// Real JSON numbers still work (direct API/CLI).
	var r2 LoadRequest
	if err := json.Unmarshal([]byte(`{"model":"m","max_num_seqs":8,"max_model_len":4096,"gpu_memory_utilization":0.9}`), &r2); err != nil {
		t.Fatalf("numeric: %v", err)
	}
	if r2.MaxNumSeqs != 8 || r2.MaxModelLen != 4096 || r2.GPUMemoryUtilization != 0.9 {
		t.Errorf("numbers not parsed: %+v", r2)
	}

	// Blank string leaves the field at its zero value (so the runtime default applies).
	var r3 LoadRequest
	if err := json.Unmarshal([]byte(`{"model":"m","max_num_seqs":""}`), &r3); err != nil {
		t.Fatalf("blank: %v", err)
	}
	if r3.MaxNumSeqs != 0 {
		t.Errorf("blank max_num_seqs should stay 0, got %d", r3.MaxNumSeqs)
	}

	// A non-numeric string is a clean error, not a panic.
	var r4 LoadRequest
	if err := json.Unmarshal([]byte(`{"model":"m","max_num_seqs":"lots"}`), &r4); err == nil {
		t.Error("expected error for non-numeric max_num_seqs")
	}
}
