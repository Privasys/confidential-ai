// Package mcpsse implements a minimal MCP-over-SSE client suitable for
// use as a long-lived session per upstream MCP server.
//
// MCP-SSE protocol (subset used by confidential-ai):
//
//   1. Client opens GET <base>/sse with Accept: text/event-stream.
//   2. Server sends an `endpoint` event whose data is the session-specific
//      POST URL (typically /messages?session=<id>). The client uses that
//      URL for every subsequent JSON-RPC request.
//   3. Client POSTs JSON-RPC 2.0 requests there. Server responds with
//      HTTP 202 immediately, then dispatches the JSON-RPC response as a
//      `message` event on the SSE stream.
//   4. Server may send `notifications/tools/list_changed` to inform the
//      client its tool catalogue is stale.
//
// One Client per upstream MCP server. Reconnects with exponential
// backoff on disconnect. Concurrent ListTools / CallTool are safe.
package mcpsse

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// Tool is the JSON-RPC `tools/list` result element. The shape mirrors
// agent.Tool so the agent.Catalog can map without re-marshalling.
type Tool struct {
	Name        string          `json:"name"`
	Description string          `json:"description"`
	InputSchema json.RawMessage `json:"inputSchema"` // MCP spec uses inputSchema (camelCase)
}

// CallResult is the JSON-RPC `tools/call` result.
type CallResult struct {
	IsError bool            `json:"isError"`
	Content []ContentBlock  `json:"content"`
	Raw     json.RawMessage `json:"-"`
}

// ContentBlock is one element of CallResult.Content.
type ContentBlock struct {
	Type string `json:"type"`           // "text" | "image" | ...
	Text string `json:"text,omitempty"` // populated when type=="text"
	// JSON for image / mime / data is not surfaced; the adapter wraps
	// the whole result block as a JSON string for the model.
}

// HeaderProvider returns a per-call header set (typically the
// `Authorization` header for the token-exchange mint). Allowed to
// return nil for "no extra headers".
type HeaderProvider func() http.Header

// Client is the persistent MCP-SSE session.
type Client struct {
	baseURL string
	client  *http.Client
	logf    func(format string, args ...any)

	cancel  context.CancelFunc
	stopped chan struct{}

	mu          sync.Mutex
	endpoint    string                          // POST URL once `endpoint` event arrives
	endpointCh  chan struct{}                   // closed when endpoint set first time
	pending     map[int64]chan *jsonrpcResponse // request id -> reply channel
	connected   atomic.Bool

	// cached tool list. Refreshed on demand and invalidated by
	// `notifications/tools/list_changed`.
	toolsMu      sync.Mutex
	toolsValid   bool
	toolsCached  []Tool
	toolsExpires time.Time

	nextID atomic.Int64
}

// New constructs a client and starts the background SSE goroutine.
func New(baseURL string, httpClient *http.Client, logf func(string, ...any)) *Client {
	if httpClient == nil {
		httpClient = &http.Client{Timeout: 0} // SSE: no overall timeout
	}
	if logf == nil {
		logf = func(string, ...any) {}
	}
	c := &Client{
		baseURL:    strings.TrimRight(baseURL, "/"),
		client:     httpClient,
		logf:       logf,
		endpointCh: make(chan struct{}),
		pending:    map[int64]chan *jsonrpcResponse{},
		stopped:    make(chan struct{}),
	}
	ctx, cancel := context.WithCancel(context.Background())
	c.cancel = cancel
	go c.runSSELoop(ctx)
	return c
}

// Close terminates the background loop. Pending calls are unblocked
// with context.Canceled.
func (c *Client) Close() {
	c.cancel()
	<-c.stopped
}

// runSSELoop holds the GET /sse stream open, reconnecting with
// exponential backoff. On every reconnect it re-issues `initialize`.
func (c *Client) runSSELoop(ctx context.Context) {
	defer close(c.stopped)
	backoff := time.Second
	const maxBackoff = 30 * time.Second
	for {
		if ctx.Err() != nil {
			return
		}
		err := c.runSSEOnce(ctx)
		if err != nil && !errors.Is(err, context.Canceled) {
			c.logf("mcpsse[%s] sse loop: %v (reconnect in %s)", c.baseURL, err, backoff)
		}
		// Reset state for the next session.
		c.connected.Store(false)
		c.mu.Lock()
		c.endpoint = ""
		c.endpointCh = make(chan struct{})
		// Drain pending calls so they fail fast on reconnect.
		for id, ch := range c.pending {
			close(ch)
			delete(c.pending, id)
		}
		c.mu.Unlock()
		c.invalidateTools()

		select {
		case <-ctx.Done():
			return
		case <-time.After(backoff):
		}
		backoff *= 2
		if backoff > maxBackoff {
			backoff = maxBackoff
		}
	}
}

func (c *Client) runSSEOnce(ctx context.Context) error {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.baseURL+"/sse", nil)
	if err != nil {
		return err
	}
	req.Header.Set("Accept", "text/event-stream")
	req.Header.Set("Cache-Control", "no-cache")
	resp, err := c.client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return fmt.Errorf("sse status %d: %s", resp.StatusCode, strings.TrimSpace(string(body)))
	}
	c.logf("mcpsse[%s] sse connected", c.baseURL)

	// Once we have the endpoint, send `initialize`.
	go func() {
		select {
		case <-c.endpointCh:
		case <-ctx.Done():
			return
		}
		if err := c.initialize(ctx); err != nil {
			c.logf("mcpsse[%s] initialize: %v", c.baseURL, err)
			return
		}
		c.connected.Store(true)
	}()

	return c.readSSEStream(ctx, resp.Body)
}

func (c *Client) readSSEStream(ctx context.Context, body io.Reader) error {
	scanner := bufio.NewScanner(body)
	scanner.Buffer(make([]byte, 64*1024), 4*1024*1024) // 4 MiB max event
	var event, data string
	flush := func() {
		if data == "" {
			return
		}
		c.dispatchEvent(event, data)
		event, data = "", ""
	}
	for scanner.Scan() {
		if ctx.Err() != nil {
			return ctx.Err()
		}
		line := scanner.Text()
		switch {
		case line == "":
			flush()
		case strings.HasPrefix(line, ":"):
			// comment / keepalive
		case strings.HasPrefix(line, "event:"):
			event = strings.TrimSpace(line[6:])
		case strings.HasPrefix(line, "data:"):
			chunk := strings.TrimPrefix(line, "data:")
			if strings.HasPrefix(chunk, " ") {
				chunk = chunk[1:]
			}
			if data == "" {
				data = chunk
			} else {
				data = data + "\n" + chunk
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return err
	}
	return io.EOF
}

func (c *Client) dispatchEvent(event, data string) {
	switch event {
	case "endpoint":
		// data is either an absolute URL or a path relative to baseURL.
		ep := data
		if !strings.HasPrefix(ep, "http://") && !strings.HasPrefix(ep, "https://") {
			ep = c.baseURL + ep
		}
		c.mu.Lock()
		first := c.endpoint == ""
		c.endpoint = ep
		ch := c.endpointCh
		c.mu.Unlock()
		if first {
			close(ch)
		}
	case "message", "":
		c.handleJSONRPC([]byte(data))
	default:
		// ignore unknown event types (forward-compat).
	}
}

func (c *Client) handleJSONRPC(payload []byte) {
	var msg struct {
		ID     *json.RawMessage `json:"id,omitempty"`
		Method string           `json:"method,omitempty"`
		Params json.RawMessage  `json:"params,omitempty"`
		Result json.RawMessage  `json:"result,omitempty"`
		Error  *jsonrpcError    `json:"error,omitempty"`
	}
	if err := json.Unmarshal(payload, &msg); err != nil {
		c.logf("mcpsse[%s] decode message: %v", c.baseURL, err)
		return
	}
	// Notification (no id) - we only act on tools/list_changed today.
	if msg.ID == nil {
		if msg.Method == "notifications/tools/list_changed" {
			c.invalidateTools()
		}
		return
	}
	// Response - route by id to pending channel.
	var id int64
	if err := json.Unmarshal(*msg.ID, &id); err != nil {
		// MCP allows string ids; we always send numeric ids so a string
		// id here means "not for us" - drop it.
		return
	}
	c.mu.Lock()
	ch, ok := c.pending[id]
	delete(c.pending, id)
	c.mu.Unlock()
	if !ok {
		return
	}
	ch <- &jsonrpcResponse{Result: msg.Result, Error: msg.Error}
	close(ch)
}

// jsonrpcResponse is what dispatchEvent hands back to the waiting caller.
type jsonrpcResponse struct {
	Result json.RawMessage
	Error  *jsonrpcError
}

type jsonrpcError struct {
	Code    int             `json:"code"`
	Message string          `json:"message"`
	Data    json.RawMessage `json:"data,omitempty"`
}

func (e jsonrpcError) String() string {
	return fmt.Sprintf("jsonrpc error %d: %s", e.Code, e.Message)
}

// initialize is the MCP handshake. We claim minimal client capabilities.
func (c *Client) initialize(ctx context.Context) error {
	params := map[string]any{
		"protocolVersion": "2025-06-18",
		"capabilities":    map[string]any{},
		"clientInfo": map[string]any{
			"name":    "privasys-confidential-ai",
			"version": "1.0",
		},
	}
	if _, err := c.call(ctx, "initialize", params, nil); err != nil {
		return err
	}
	// Notify server we're ready.
	return c.notify(ctx, "notifications/initialized", nil)
}

// ListTools returns the catalogue. Cached for ttl on success and
// invalidated by tools/list_changed notifications.
func (c *Client) ListTools(ctx context.Context, ttl time.Duration) ([]Tool, error) {
	c.toolsMu.Lock()
	if c.toolsValid && time.Now().Before(c.toolsExpires) {
		out := c.toolsCached
		c.toolsMu.Unlock()
		return out, nil
	}
	c.toolsMu.Unlock()

	if err := c.waitConnected(ctx); err != nil {
		return nil, err
	}
	raw, err := c.call(ctx, "tools/list", map[string]any{}, nil)
	if err != nil {
		return nil, err
	}
	var payload struct {
		Tools []Tool `json:"tools"`
	}
	if err := json.Unmarshal(raw, &payload); err != nil {
		return nil, err
	}
	c.toolsMu.Lock()
	c.toolsCached = payload.Tools
	c.toolsValid = true
	c.toolsExpires = time.Now().Add(ttl)
	c.toolsMu.Unlock()
	return payload.Tools, nil
}

// CallTool executes one MCP `tools/call`.
func (c *Client) CallTool(ctx context.Context, name string, args json.RawMessage, headers HeaderProvider) (*CallResult, error) {
	if err := c.waitConnected(ctx); err != nil {
		return nil, err
	}
	if len(args) == 0 {
		args = []byte(`{}`)
	}
	params := map[string]any{
		"name":      name,
		"arguments": json.RawMessage(args),
	}
	raw, err := c.call(ctx, "tools/call", params, headers)
	if err != nil {
		return nil, err
	}
	res := &CallResult{Raw: raw}
	if err := json.Unmarshal(raw, res); err != nil {
		// Server returned a non-standard payload; surface it raw.
		return res, nil
	}
	return res, nil
}

// invalidateTools forces the next ListTools to refetch.
func (c *Client) invalidateTools() {
	c.toolsMu.Lock()
	c.toolsValid = false
	c.toolsCached = nil
	c.toolsMu.Unlock()
}

// waitConnected blocks until the SSE handshake (endpoint + initialize)
// finishes, or ctx expires.
func (c *Client) waitConnected(ctx context.Context) error {
	if c.connected.Load() {
		return nil
	}
	deadline := time.NewTimer(5 * time.Second)
	defer deadline.Stop()
	tick := time.NewTicker(50 * time.Millisecond)
	defer tick.Stop()
	for {
		if c.connected.Load() {
			return nil
		}
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-deadline.C:
			return errors.New("mcp_sse: timed out waiting for handshake")
		case <-tick.C:
		}
	}
}

// call sends a JSON-RPC request and waits for the response.
func (c *Client) call(ctx context.Context, method string, params any, headers HeaderProvider) (json.RawMessage, error) {
	id := c.nextID.Add(1)
	req := map[string]any{
		"jsonrpc": "2.0",
		"id":      id,
		"method":  method,
		"params":  params,
	}
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	c.mu.Lock()
	endpoint := c.endpoint
	if endpoint == "" {
		c.mu.Unlock()
		// Wait for endpoint event.
		select {
		case <-c.endpointCh:
			c.mu.Lock()
			endpoint = c.endpoint
			c.mu.Unlock()
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(5 * time.Second):
			return nil, errors.New("mcp_sse: no endpoint event from server")
		}
	} else {
		c.mu.Unlock()
	}

	ch := make(chan *jsonrpcResponse, 1)
	c.mu.Lock()
	c.pending[id] = ch
	c.mu.Unlock()
	defer func() {
		c.mu.Lock()
		delete(c.pending, id)
		c.mu.Unlock()
	}()

	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	if headers != nil {
		if extra := headers(); extra != nil {
			for k, vs := range extra {
				for _, v := range vs {
					httpReq.Header.Add(k, v)
				}
			}
		}
	}
	postClient := &http.Client{Timeout: 30 * time.Second}
	resp, err := postClient.Do(httpReq)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	if resp.StatusCode >= 300 {
		body, _ := io.ReadAll(io.LimitReader(resp.Body, 4096))
		return nil, fmt.Errorf("mcp_sse post %s: status %d: %s",
			method, resp.StatusCode, strings.TrimSpace(string(body)))
	}
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))

	select {
	case rep, ok := <-ch:
		if !ok || rep == nil {
			return nil, errors.New("mcp_sse: connection lost waiting for response")
		}
		if rep.Error != nil {
			return nil, errors.New(rep.Error.String())
		}
		return rep.Result, nil
	case <-ctx.Done():
		return nil, ctx.Err()
	}
}

// notify sends a JSON-RPC notification (no response expected).
func (c *Client) notify(ctx context.Context, method string, params any) error {
	req := map[string]any{
		"jsonrpc": "2.0",
		"method":  method,
		"params":  params,
	}
	body, _ := json.Marshal(req)

	c.mu.Lock()
	endpoint := c.endpoint
	c.mu.Unlock()
	if endpoint == "" {
		return errors.New("mcp_sse: not connected")
	}
	httpReq, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(body))
	if err != nil {
		return err
	}
	httpReq.Header.Set("Content-Type", "application/json")
	postClient := &http.Client{Timeout: 5 * time.Second}
	resp, err := postClient.Do(httpReq)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, io.LimitReader(resp.Body, 4096))
	return nil
}
