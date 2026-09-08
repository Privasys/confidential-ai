package billing

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// A caller with no billing account is refused, never served on the owner's
// tab (Bertrand, 2026-09-08). The management-service answers
// GET <report base>/billable-caller?sub=<caller> behind the same usage-only
// bearer the reporter pushes with; the enclave asks once per caller and
// caches the verdict, so the hot path stays a map lookup. Only the pairwise
// subject travels — never a prompt — and the answer is a bare verdict.
const (
	billableTTL   = 15 * time.Minute // a known-good caller is re-checked rarely
	unbillableTTL = time.Minute      // a refused caller can open an account and retry soon
)

type billableEntry struct {
	billable bool
	until    time.Time
}

// ErrBillableUnknown is returned when the management-service could not be
// asked (network, 5xx) and no cached verdict exists. The caller decides the
// failure mode; inference fails open on it, consistent with the freeze probe.
var ErrBillableUnknown = errors.New("billing: caller billability unknown")

// billableURL derives the check endpoint from the configured report URL:
// ".../enclave/ai-usage" → ".../enclave/billable-caller".
func (r *Reporter) billableURL(sub string) string {
	base := strings.TrimSuffix(strings.TrimRight(r.cfg.ReportURL, "/"), "/ai-usage")
	return base + "/billable-caller?sub=" + url.QueryEscape(sub)
}

// CallerBillable reports whether sub maps to an account the platform can
// debit. A nil Reporter (metering off) cannot know and answers true. On a
// definitive answer the verdict is cached; on a transport failure the last
// cached verdict (even expired) is reused, else ErrBillableUnknown.
func (r *Reporter) CallerBillable(ctx context.Context, sub string) (bool, error) {
	if r == nil || sub == "" {
		return true, nil
	}
	now := time.Now()
	r.billableMu.Lock()
	if r.billable == nil {
		r.billable = map[string]billableEntry{}
	}
	e, cached := r.billable[sub]
	r.billableMu.Unlock()
	if cached && now.Before(e.until) {
		return e.billable, nil
	}

	verdict, err := r.askBillable(ctx, sub)
	if err != nil {
		if cached {
			return e.billable, nil // stale verdict beats an outage
		}
		return false, fmt.Errorf("%w: %v", ErrBillableUnknown, err)
	}
	ttl := billableTTL
	if !verdict {
		ttl = unbillableTTL
	}
	r.billableMu.Lock()
	r.billable[sub] = billableEntry{billable: verdict, until: now.Add(ttl)}
	r.billableMu.Unlock()
	return verdict, nil
}

func (r *Reporter) askBillable(ctx context.Context, sub string) (bool, error) {
	reqCtx, cancel := context.WithTimeout(ctx, 5*time.Second)
	defer cancel()
	req, err := http.NewRequestWithContext(reqCtx, http.MethodGet, r.billableURL(sub), nil)
	if err != nil {
		return false, err
	}
	if r.cfg.ReportToken != "" {
		req.Header.Set("Authorization", "Bearer "+r.cfg.ReportToken)
	}
	resp, err := r.client.Do(req)
	if err != nil {
		return false, err
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		return false, fmt.Errorf("billable-caller status %d", resp.StatusCode)
	}
	var out struct {
		Billable bool `json:"billable"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		return false, err
	}
	return out.Billable, nil
}
