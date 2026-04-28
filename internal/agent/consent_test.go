package agent

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestConsentResolveAllow(t *testing.T) {
	reg := NewConsentRegistry()
	go func() {
		time.Sleep(20 * time.Millisecond)
		if !reg.Resolve("call-1", ConsentDecision{Allowed: true}) {
			t.Errorf("Resolve returned false")
		}
	}()
	d, err := reg.Wait(context.Background(), "call-1", time.Second)
	if err != nil {
		t.Fatalf("Wait err: %v", err)
	}
	if !d.Allowed {
		t.Fatalf("expected allowed=true, got %+v", d)
	}
	if reg.Pending() != 0 {
		t.Fatalf("registry should be empty, got %d", reg.Pending())
	}
}

func TestConsentResolveDeny(t *testing.T) {
	reg := NewConsentRegistry()
	go func() {
		time.Sleep(20 * time.Millisecond)
		reg.Resolve("call-2", ConsentDecision{Allowed: false, Reason: "nope"})
	}()
	d, err := reg.Wait(context.Background(), "call-2", time.Second)
	if err != nil {
		t.Fatalf("Wait err: %v", err)
	}
	if d.Allowed || d.Reason != "nope" {
		t.Fatalf("unexpected decision %+v", d)
	}
}

func TestConsentTimeout(t *testing.T) {
	reg := NewConsentRegistry()
	_, err := reg.Wait(context.Background(), "call-3", 30*time.Millisecond)
	if !errors.Is(err, ErrConsentTimeout) {
		t.Fatalf("expected ErrConsentTimeout, got %v", err)
	}
	if reg.Pending() != 0 {
		t.Fatalf("registry should be empty, got %d", reg.Pending())
	}
}

func TestConsentContextCancel(t *testing.T) {
	reg := NewConsentRegistry()
	ctx, cancel := context.WithCancel(context.Background())
	go func() {
		time.Sleep(20 * time.Millisecond)
		cancel()
	}()
	_, err := reg.Wait(ctx, "call-4", time.Second)
	if !errors.Is(err, context.Canceled) {
		t.Fatalf("expected context.Canceled, got %v", err)
	}
}

func TestConsentResolveUnknown(t *testing.T) {
	reg := NewConsentRegistry()
	if reg.Resolve("ghost", ConsentDecision{Allowed: true}) {
		t.Fatalf("Resolve on unknown id should return false")
	}
}
