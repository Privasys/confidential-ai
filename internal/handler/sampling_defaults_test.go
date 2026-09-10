// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"encoding/json"
	"testing"

	"github.com/privasys/confidential-ai/internal/models"
)

// A request that omits the sampling fields is forwarded with the model's
// defaults spelled out and the block reports those same values; a
// request that sets them keeps its own values.
func TestInjectSamplingDefaults(t *testing.T) {
	d := models.GenerationDefaults{Temperature: 1.0, TopP: 0.95, TopK: 20}

	out, eff, err := injectSamplingDefaults([]byte(`{"model":"m","messages":[]}`), d)
	if err != nil {
		t.Fatal(err)
	}
	var m map[string]any
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["temperature"] != 1.0 || m["top_p"] != 0.95 || m["top_k"] != 20.0 {
		t.Fatalf("defaults not injected: %v", m)
	}
	if eff != d {
		t.Fatalf("effective %+v, want %+v", eff, d)
	}

	out, eff, err = injectSamplingDefaults([]byte(`{"model":"m","temperature":0.2,"top_p":1,"top_k":-1}`), d)
	if err != nil {
		t.Fatal(err)
	}
	if err := json.Unmarshal(out, &m); err != nil {
		t.Fatal(err)
	}
	if m["temperature"] != 0.2 || m["top_p"] != 1.0 || m["top_k"] != -1.0 {
		t.Fatalf("client values overridden: %v", m)
	}
	if eff.Temperature != 0.2 || eff.TopP != 1.0 || eff.TopK != -1 {
		t.Fatalf("effective %+v does not reflect the client's values", eff)
	}
}
