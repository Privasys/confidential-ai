// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"encoding/json"
	"testing"
)

func TestWatermarkingIsOptIn(t *testing.T) {
	for _, c := range []struct {
		body string
		want bool
	}{
		{`{"model":"m"}`, false},
		{`{"model":"m","watermarking":null}`, false},
		{`{"model":"m","watermarking":false}`, false},
		{`{"model":"m","watermarking":true}`, true},
	} {
		out, want, err := optInWatermarking([]byte(c.body))
		if err != nil {
			t.Fatal(err)
		}
		if want != c.want {
			t.Fatalf("%s: asked for the mark = %v, want %v", c.body, want, c.want)
		}
		var m map[string]any
		if err := json.Unmarshal(out, &m); err != nil {
			t.Fatal(err)
		}
		if m["watermarking"] != c.want {
			t.Fatalf("%s: forwarded watermarking = %v, want %v", c.body, m["watermarking"], c.want)
		}
	}
}
