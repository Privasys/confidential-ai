// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package handler

import (
	"bytes"
	"encoding/json"
	"io"
	"net/http"
	"time"

	"github.com/privasys/confidential-ai/internal/models"
)

// maxDetectText bounds a detection request. Scoring is CPU work on the
// tokens, and a few pages are far more than a p-value needs.
const maxDetectText = 64 << 10

// watermarkDetect handles POST /v1/watermark/detect: does a text carry this
// instance's watermark? The answer is a p-value from the loopback detection
// helper, which holds the key; the key never leaves the enclave. Callers are
// authorised like inference. The check is free but not public: an
// unlimited oracle helps whoever tries to strip or forge the mark.
//
// Request:  {"text": "..."}
// Response: {"watermark": "<key id>", "is_watermarked": bool, "p_value": f,
//
//	"score": f, "num_scored_tokens": n}
func (h *Handler) watermarkDetect(w http.ResponseWriter, r *http.Request) {
	if _, ok := h.authorizeInference(w, r); !ok {
		return
	}
	keyID := ""
	if h.fleet != nil && h.fleet.Generate() != nil {
		keyID = h.fleet.Generate().WatermarkKeyID()
	}
	if keyID == "" {
		writeError(w, http.StatusNotFound, "this instance does not serve with a watermark")
		return
	}
	body, err := io.ReadAll(io.LimitReader(r.Body, maxDetectText+4<<10))
	if err != nil {
		writeError(w, http.StatusBadRequest, "failed to read request")
		return
	}
	var req struct {
		Text string `json:"text"`
	}
	if err := json.Unmarshal(body, &req); err != nil || req.Text == "" {
		writeError(w, http.StatusBadRequest, `body must be {"text": "..."}`)
		return
	}
	if len(req.Text) > maxDetectText {
		writeError(w, http.StatusRequestEntityTooLarge, "text is longer than 64 KiB")
		return
	}
	fwd, _ := json.Marshal(map[string]string{"text": req.Text})
	client := &http.Client{Timeout: 60 * time.Second}
	resp, err := client.Post(models.WatermarkDetectorURL()+"/detect", "application/json", bytes.NewReader(fwd))
	if err != nil {
		writeError(w, http.StatusServiceUnavailable, "watermark detector not ready")
		return
	}
	defer resp.Body.Close()
	var out map[string]any
	if resp.StatusCode != http.StatusOK || json.NewDecoder(io.LimitReader(resp.Body, 64<<10)).Decode(&out) != nil {
		writeError(w, http.StatusBadGateway, "watermark detector failed")
		return
	}
	out["watermark"] = keyID
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(out)
}
