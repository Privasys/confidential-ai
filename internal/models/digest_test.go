// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package models

import (
	"context"
	"crypto/sha256"
	"encoding/hex"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func sha(b string) string { s := sha256.Sum256([]byte(b)); return hex.EncodeToString(s[:]) }

// fakeHFRepo lays out a Hugging Face cache the way huggingface_hub does:
// LFS files as blobs named by their SHA-256, symlinked from a snapshot that
// refs/main points at; small files as blobs named by something else.
func fakeHFRepo(t *testing.T, hub, repo string, weights string) string {
	t.Helper()
	dir := filepath.Join(hub, "models--"+strings.ReplaceAll(repo, "/", "--"))
	commit := "0123456789abcdef0123456789abcdef01234567"
	snap := filepath.Join(dir, "snapshots", commit)
	blobs := filepath.Join(dir, "blobs")
	for _, d := range []string{snap, blobs, filepath.Join(dir, "refs")} {
		if err := os.MkdirAll(d, 0o755); err != nil {
			t.Fatal(err)
		}
	}
	os.WriteFile(filepath.Join(dir, "refs", "main"), []byte(commit), 0o644)
	put := func(name, blob, content string) {
		os.WriteFile(filepath.Join(blobs, blob), []byte(content), 0o644)
		if err := os.Symlink(filepath.Join("..", "..", "blobs", blob), filepath.Join(snap, name)); err != nil {
			t.Skipf("symlinks unavailable: %v", err)
		}
	}
	put("model.safetensors", sha(weights), weights)
	put("config.json", "3f1c0a2b9e", `{"model_type":"qwen3"}`)
	put("generation_config.json", "8d2e7f", `{"temperature":0.6,"top_p":0.95,"top_k":20}`)
	return snap
}

func TestDigestCoversEveryFileFromTheHFCache(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	snap := fakeHFRepo(t, hub, "Org/Model", "weights-v1")

	dir := resolveModelDir("Org/Model")
	if dir != snap {
		t.Fatalf("resolved %q, want the refs/main snapshot %q", dir, snap)
	}
	entries, err := modelManifest(dir)
	if err != nil {
		t.Fatal(err)
	}
	if len(entries) != 3 {
		t.Fatalf("want 3 files, got %+v", entries)
	}
	// sha256sum format, sorted by path: what `find | sort | xargs sha256sum | sha256sum` prints.
	want := sha(sha(`{"model_type":"qwen3"}`) + "  config.json\n" +
		sha(`{"temperature":0.6,"top_p":0.95,"top_k":20}`) + "  generation_config.json\n" +
		sha("weights-v1") + "  model.safetensors\n")
	if got := manifestDigest(entries); got != want {
		t.Fatalf("digest %s, want %s", got, want)
	}
	if _, err := os.Stat(filepath.Join(verifiedDir(), sha("weights-v1"))); err != nil {
		t.Fatalf("verified blob not recorded: %v", err)
	}
	if d := loadGenerationDefaults(dir); d.Temperature != 0.6 || d.TopP != 0.95 || d.TopK != 20 {
		t.Fatalf("generation defaults not read from the snapshot: %+v", d)
	}

	// Different weights, different digest.
	hub2 := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub2)
	fakeHFRepo(t, hub2, "Org/Model", "weights-v2")
	e2, err := modelManifest(resolveModelDir("Org/Model"))
	if err != nil {
		t.Fatal(err)
	}
	if manifestDigest(e2) == want {
		t.Fatal("a weights change must change the digest")
	}
}

func TestTamperedBlobFailsTheDigest(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	snap := fakeHFRepo(t, hub, "Org/Model", "weights-v1")
	target, _ := filepath.EvalSymlinks(filepath.Join(snap, "model.safetensors"))
	os.WriteFile(target, []byte("weights-swapped"), 0o644)

	if _, err := modelManifest(snap); err == nil || !strings.Contains(err.Error(), "content hashes to") {
		t.Fatalf("a blob whose content does not match its name must fail, got %v", err)
	}
	if _, err := os.Stat(filepath.Join(verifiedDir(), sha("weights-v1"))); err == nil {
		t.Fatal("a mismatching blob must not be recorded as verified")
	}
}

func TestAmbiguousSnapshotsAreNotGuessed(t *testing.T) {
	hub := t.TempDir()
	t.Setenv("HF_HUB_CACHE", hub)
	dir := filepath.Join(hub, "models--Org--Model", "snapshots")
	os.MkdirAll(filepath.Join(dir, "aaaa"), 0o755)
	os.MkdirAll(filepath.Join(dir, "bbbb"), 0o755)
	if got := resolveModelDir("Org/Model"); got != "" {
		t.Fatalf("two snapshots and no main ref must resolve to nothing, got %q", got)
	}
}

func TestSnapshotRevision(t *testing.T) {
	for in, want := range map[string]string{
		"/data/.cache/huggingface/hub/models--Qwen--X/snapshots/c653bba46dfa": "c653bba46dfa",
		"/models/qwen36-35b-a3b-fp8":                                          "",
		"":                                                                    "",
	} {
		if got := snapshotRevision(in); got != want {
			t.Fatalf("%q: got %q, want %q", in, got, want)
		}
	}
}

func TestFetchFullSnapshotSkipsLocalDirectories(t *testing.T) {
	for _, p := range []string{"/models/qwen", "qwen36-35b-a3b-fp8"} {
		if err := fetchFullSnapshot(context.Background(), p); err != nil {
			t.Fatalf("%s: a local model must not be fetched: %v", p, err)
		}
	}
}
