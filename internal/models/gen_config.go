// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package models

import (
	"encoding/json"
	"os"
	"path/filepath"
)

// GenerationDefaults are the sampling values vLLM applies when a request
// omits them: the model's generation_config.json (vLLM's default
// --generation-config=auto), falling back to vLLM's own defaults.
//
// They matter for reproducibility: a request without temperature/top_p/
// top_k is sampled with THESE values, so they must be made explicit on the
// forwarded request and reported in the reproducibility block, or a replay
// that pins "what the block said" samples differently. Found 2026-09-10:
// Qwen3.6 ships temperature 1.0, top_p 0.95, top_k 20; the block reported
// 1 / 1 / none, and every pinned replay of a client that omitted them
// diverged from its original while the engine was provably deterministic.
type GenerationDefaults struct {
	Temperature float64 `json:"temperature"`
	TopP        float64 `json:"top_p"`
	TopK        int     `json:"top_k"`
}

// vllmGenerationDefaults are vLLM's values when neither the request nor
// the model's generation_config.json sets a field.
var vllmGenerationDefaults = GenerationDefaults{Temperature: 1.0, TopP: 1.0, TopK: -1}

// loadGenerationDefaults reads <modelPath>/generation_config.json. A
// missing or unreadable file yields vLLM's defaults; a present file
// overrides only the fields it carries (a generation config may set
// top_k without top_p, and so on).
func loadGenerationDefaults(modelPath string) GenerationDefaults {
	d := vllmGenerationDefaults
	if modelPath == "" {
		return d
	}
	raw, err := os.ReadFile(filepath.Join(modelPath, "generation_config.json"))
	if err != nil {
		return d
	}
	var cfg struct {
		Temperature *float64 `json:"temperature"`
		TopP        *float64 `json:"top_p"`
		TopK        *int     `json:"top_k"`
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return d
	}
	if cfg.Temperature != nil {
		d.Temperature = *cfg.Temperature
	}
	if cfg.TopP != nil {
		d.TopP = *cfg.TopP
	}
	if cfg.TopK != nil {
		d.TopK = *cfg.TopK
	}
	return d
}

// GenerationDefaults returns the sampling defaults of the served model
// (vLLM's own when no model is loaded).
func (m *Manager) GenerationDefaults() GenerationDefaults {
	m.mu.RLock()
	defer m.mu.RUnlock()
	if m.genDefaults == nil {
		return vllmGenerationDefaults
	}
	return *m.genDefaults
}
