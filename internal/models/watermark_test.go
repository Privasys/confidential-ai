// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package models

import (
	"os"
	"strconv"
	"strings"
	"testing"
)

func TestInstanceWatermarkKeyIsCreatedOnceAndKeptPrivate(t *testing.T) {
	t.Setenv("HOME", t.TempDir()) // vllmHome falls back to HOME without /data

	k1, err := instanceWatermarkKey()
	if err != nil {
		t.Fatal(err)
	}
	k2, err := instanceWatermarkKey()
	if err != nil {
		t.Fatal(err)
	}
	if k1 != k2 {
		t.Fatalf("the key changed between loads: %d then %d", k1, k2)
	}
	info, err := os.Stat(watermarkKeyPath())
	if err != nil {
		t.Fatal(err)
	}
	if perm := info.Mode().Perm(); perm != 0o600 {
		t.Fatalf("key file mode %o, want 600", perm)
	}
}

func TestWatermarkKeyIDDoesNotRevealTheKey(t *testing.T) {
	key := uint64(1234567890123456789)
	id := watermarkKeyID(key)
	if len(id) != 16 || strings.Contains(id, strconv.FormatUint(key, 10)) {
		t.Fatalf("id %q", id)
	}
	if id != watermarkKeyID(key) || id == watermarkKeyID(key+1) {
		t.Fatal("the id must be stable per key and differ between keys")
	}
}

func TestEngineOutputIsRedactedOfTheKey(t *testing.T) {
	key := uint64(98765432109876543)
	arg := watermarkConfigArg(key)
	line := "INFO non-default args: {'watermark_config': '" + arg + "'}"
	got := redactWatermarkKey(line, strconv.FormatUint(key, 10))
	if strings.Contains(got, strconv.FormatUint(key, 10)) {
		t.Fatalf("key leaked: %s", got)
	}
	if redactWatermarkKey("no key here", "") != "no key here" {
		t.Fatal("an unwatermarked load must leave lines alone")
	}
}
