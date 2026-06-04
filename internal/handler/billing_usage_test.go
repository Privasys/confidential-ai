package handler

import (
	"encoding/json"
	"testing"
)

func TestExtractUsage(t *testing.T) {
	body := []byte(`{"id":"chatcmpl-abc","object":"chat.completion","usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}`)
	id, in, out, ok := extractUsage(body)
	if !ok || id != "chatcmpl-abc" || in != 12 || out != 34 {
		t.Fatalf("extractUsage = (%q,%d,%d,%v), want (chatcmpl-abc,12,34,true)", id, in, out, ok)
	}

	// No usage block (e.g. an error body) -> not metered.
	if _, _, _, ok := extractUsage([]byte(`{"error":"boom"}`)); ok {
		t.Fatal("extractUsage on error body should return ok=false")
	}
	if _, _, _, ok := extractUsage([]byte(`not json`)); ok {
		t.Fatal("extractUsage on garbage should return ok=false")
	}
}

func TestExtractStreamUsage(t *testing.T) {
	// The include_usage final chunk: empty choices, populated usage.
	usageChunk := []byte("data: {\"id\":\"chatcmpl-xyz\",\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":21}}\n\n")
	id, in, out, ok := extractStreamUsage(usageChunk)
	if !ok || id != "chatcmpl-xyz" || in != 7 || out != 21 {
		t.Fatalf("extractStreamUsage = (%q,%d,%d,%v), want (chatcmpl-xyz,7,21,true)", id, in, out, ok)
	}

	// An ordinary delta chunk has no usage -> not a usage frame.
	delta := []byte("data: {\"id\":\"chatcmpl-xyz\",\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n")
	if _, _, _, ok := extractStreamUsage(delta); ok {
		t.Fatal("extractStreamUsage on a delta chunk should return ok=false")
	}

	// The [DONE] sentinel is not a usage frame.
	if _, _, _, ok := extractStreamUsage([]byte("data: [DONE]\n\n")); ok {
		t.Fatal("extractStreamUsage on [DONE] should return ok=false")
	}
}

func TestInjectStreamUsage(t *testing.T) {
	// Client did not set stream_options: we add include_usage and report it.
	out, clientHad := injectStreamUsage([]byte(`{"model":"m","stream":true}`))
	if clientHad {
		t.Fatal("clientHad should be false when client omitted stream_options")
	}
	var got map[string]any
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("injected body is not valid JSON: %v", err)
	}
	opts, _ := got["stream_options"].(map[string]any)
	if v, _ := opts["include_usage"].(bool); !v {
		t.Fatalf("expected stream_options.include_usage=true, got %v", got["stream_options"])
	}
	// Re-injecting an already-opted-in body reports clientHad=true.
	_, clientHad = injectStreamUsage([]byte(`{"model":"m","stream":true,"stream_options":{"include_usage":true}}`))
	if !clientHad {
		t.Fatal("clientHad should be true when client set include_usage")
	}
	// Garbage in -> returned unchanged, clientHad=false.
	in := []byte("not json")
	gotBytes, clientHad := injectStreamUsage(in)
	if clientHad || string(gotBytes) != string(in) {
		t.Fatal("injectStreamUsage on garbage should return body unchanged, clientHad=false")
	}
}
