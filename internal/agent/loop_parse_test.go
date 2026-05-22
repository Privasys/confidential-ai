package agent

import (
	"encoding/json"
	"testing"
)

// TestParseToolCalls_StructuredField covers the happy path: vLLM
// populated `tool_calls` natively (e.g. tool_choice:"auto" + matching
// per-model parser).
func TestParseToolCalls_StructuredField(t *testing.T) {
	body := []byte(`{
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "",
	      "tool_calls": [{
	        "id": "call_1",
	        "type": "function",
	        "function": {"name": "brave_search__search", "arguments": "{\"query\":\"hi\"}"}
	      }]
	    }
	  }]
	}`)
	calls, _, err := parseToolCalls(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "brave_search__search" {
		t.Fatalf("unexpected calls: %#v", calls)
	}
}

// TestParseToolCalls_RescuedFromContentArray covers the regression: vLLM
// 0.21 with tool_choice:"required" emits the call as a JSON array
// embedded in `content` and leaves `tool_calls` empty / unset.
func TestParseToolCalls_RescuedFromContentArray(t *testing.T) {
	body := []byte(`{
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "[{\"name\":\"brave_search__search\",\"parameters\":{\"count\":10,\"query\":\"site:privasys.org enclave\"}}]"
	    }
	  }]
	}`)
	calls, msg, err := parseToolCalls(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(calls) != 1 {
		t.Fatalf("expected 1 rescued call, got %d", len(calls))
	}
	if calls[0].Function.Name != "brave_search__search" {
		t.Fatalf("name: %s", calls[0].Function.Name)
	}
	var args map[string]any
	if err := json.Unmarshal(calls[0].Function.Arguments, &args); err != nil {
		t.Fatalf("args not valid JSON: %v", err)
	}
	if args["query"] != "site:privasys.org enclave" {
		t.Fatalf("args: %#v", args)
	}
	// content must be cleared and tool_calls mirrored so downstream
	// stream-replay logic sees the rescued call.
	if msg["content"] != "" {
		t.Fatalf("content not cleared: %v", msg["content"])
	}
	if tc, _ := msg["tool_calls"].([]any); len(tc) != 1 {
		t.Fatalf("tool_calls not mirrored: %v", msg["tool_calls"])
	}
}

// TestParseToolCalls_RescuedFromContentSingleObject covers the
// degenerate case where the model emits a single object rather than a
// one-element array, with the OpenAI-style `arguments` key.
func TestParseToolCalls_RescuedFromContentSingleObject(t *testing.T) {
	body := []byte(`{
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "{\"name\":\"get_weather\",\"arguments\":{\"city\":\"Paris\"}}"
	    }
	  }]
	}`)
	calls, _, err := parseToolCalls(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(calls) != 1 || calls[0].Function.Name != "get_weather" {
		t.Fatalf("calls: %#v", calls)
	}
}

// TestParseToolCalls_NaturalLanguageContentIgnored ensures the
// fallback regex does not misfire on prose containing braces.
func TestParseToolCalls_NaturalLanguageContentIgnored(t *testing.T) {
	body := []byte(`{
	  "choices": [{
	    "message": {
	      "role": "assistant",
	      "content": "I don't need any tools to answer that."
	    }
	  }]
	}`)
	calls, _, err := parseToolCalls(body)
	if err != nil {
		t.Fatalf("err: %v", err)
	}
	if len(calls) != 0 {
		t.Fatalf("unexpected calls: %#v", calls)
	}
}
