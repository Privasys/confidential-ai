// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"strings"
	"testing"
)

func TestEngineLinesKeepOnlyVLLMFamilies(t *testing.T) {
	in := strings.Join([]string{
		"# HELP python_gc_objects_collected_total Objects collected during gc",
		"# TYPE python_gc_objects_collected_total counter",
		`python_gc_objects_collected_total{generation="0"} 123.0`,
		"# HELP vllm:prefix_cache_hits_total Prefix cache hits, in terms of number of cached tokens.",
		"# TYPE vllm:prefix_cache_hits_total counter",
		`vllm:prefix_cache_hits_total{engine="0",model_name="qwen36-35b-a3b-fp8"} 4096.0`,
		`vllm:kv_cache_usage_perc{engine="0",model_name="qwen36-35b-a3b-fp8"} 0.25`,
		"process_resident_memory_bytes 1.0e+09",
	}, "\n")
	var out strings.Builder
	writeEngineLines(&out, strings.NewReader(in))
	got := out.String()
	for _, want := range []string{
		"# HELP vllm:prefix_cache_hits_total",
		"# TYPE vllm:prefix_cache_hits_total counter",
		`vllm:prefix_cache_hits_total{engine="0",model_name="qwen36-35b-a3b-fp8"} 4096.0`,
		`vllm:kv_cache_usage_perc{engine="0",model_name="qwen36-35b-a3b-fp8"} 0.25`,
	} {
		if !strings.Contains(got, want) {
			t.Errorf("missing %q in:\n%s", want, got)
		}
	}
	for _, drop := range []string{"python_gc", "process_resident"} {
		if strings.Contains(got, drop) {
			t.Errorf("%s leaked into:\n%s", drop, got)
		}
	}
}
