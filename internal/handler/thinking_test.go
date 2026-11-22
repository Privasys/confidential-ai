// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"encoding/json"
	"testing"
)

func TestBudgetlessThinkingBecomesAdaptive(t *testing.T) {
	for _, c := range []struct{ in, wantType string }{
		{`{"thinking":{"type":"enabled"}}`, "adaptive"},
		{`{"thinking":{"type":"enabled","budget_tokens":null}}`, "adaptive"},
		{`{"thinking":{"type":"enabled","budget_tokens":2048}}`, "enabled"},
		{`{"thinking":{"type":"disabled"}}`, "disabled"},
		{`{"thinking":{"type":"adaptive"}}`, "adaptive"},
	} {
		out, err := budgetlessThinkingToAdaptive([]byte(c.in))
		if err != nil {
			t.Fatal(err)
		}
		var m struct {
			Thinking map[string]any `json:"thinking"`
		}
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		if m.Thinking["type"] != c.wantType {
			t.Fatalf("%s: type %v, want %s", c.in, m.Thinking["type"], c.wantType)
		}
	}
	if out, _ := budgetlessThinkingToAdaptive([]byte(`{"model":"m"}`)); string(out) != `{"model":"m"}` {
		t.Fatalf("no thinking field must pass through unchanged: %s", out)
	}
}
