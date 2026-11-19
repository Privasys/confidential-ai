// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package models

import (
	"crypto/sha256"
	"encoding/hex"
	"fmt"
	"io"
	"io/fs"
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"strings"
	"sync"
)

// The model digest is what a client pins to know which weights answer it.
// It must cover every byte vLLM reads, and a third party must be able to
// recompute it from the publisher's files without trusting this enclave.
//
// For a model without a dm-verity disk it is the SHA-256 of a manifest in
// `sha256sum` format: one line per file, "<sha256>  <relative path>\n",
// sorted by path. The same value comes out of
//
//	cd <snapshot> && find -L . -type f ! -name '.*' | sed 's|^\./||' | LC_ALL=C sort | xargs sha256sum | sha256sum
//
// and every line's hash can be checked against the SHA-256 the model's
// publisher lists for that file (Hugging Face shows it per LFS file). The
// manifest itself is published in /.well-known/served-models.

// ManifestEntry is one file of a model's digest manifest.
type ManifestEntry struct {
	Path   string `json:"path"`
	SHA256 string `json:"sha256"`
	Size   int64  `json:"size"`
}

// vllmHome is the HOME the vLLM processes run under. On an app with an
// encrypted volume it is /data, so the Hugging Face, Triton and FlashInfer
// caches survive restarts inside the volume. The digest and the generation
// defaults must read the model from the same place vLLM downloaded it to.
func vllmHome() string {
	if info, err := os.Stat("/data"); err == nil && info.IsDir() {
		return "/data"
	}
	home, _ := os.UserHomeDir()
	return home
}

// hfHubDir is the Hugging Face hub cache vLLM downloads into.
func hfHubDir() string {
	if v := os.Getenv("HF_HUB_CACHE"); v != "" {
		return v
	}
	if v := os.Getenv("HF_HOME"); v != "" {
		return filepath.Join(v, "hub")
	}
	return filepath.Join(vllmHome(), ".cache", "huggingface", "hub")
}

// resolveModelDir returns the directory holding a model's files: the path
// itself when it is a directory, otherwise the Hugging Face snapshot the
// repository's main ref points at. Empty when neither exists.
func resolveModelDir(modelPath string) string {
	if info, err := os.Stat(modelPath); err == nil && info.IsDir() {
		return modelPath
	}
	if !strings.Contains(modelPath, "/") || filepath.IsAbs(modelPath) {
		return ""
	}
	repo := filepath.Join(hfHubDir(), "models--"+strings.ReplaceAll(modelPath, "/", "--"))
	if ref, err := os.ReadFile(filepath.Join(repo, "refs", "main")); err == nil {
		snap := filepath.Join(repo, "snapshots", strings.TrimSpace(string(ref)))
		if info, err := os.Stat(snap); err == nil && info.IsDir() {
			return snap
		}
	}
	// No main ref (a pinned revision): exactly one snapshot is unambiguous,
	// several are not, and guessing would attest the wrong weights.
	entries, err := os.ReadDir(filepath.Join(repo, "snapshots"))
	if err != nil || len(entries) != 1 {
		return ""
	}
	return filepath.Join(repo, "snapshots", entries[0].Name())
}

// verifiedDir records, on the encrypted volume, the Hugging Face blobs
// whose content was hashed and found equal to their name, so a 35 GB model
// is read for its digest once per download rather than at every boot. The
// volume is integrity-protected, so a marker cannot be forged from outside.
func verifiedDir() string { return filepath.Join(filepath.Dir(hfHubDir()), "privasys-verified") }

// isSHA256Hex reports whether s is a lowercase 64-character hex digest.
func isSHA256Hex(s string) bool {
	if len(s) != 64 {
		return false
	}
	for _, c := range s {
		if !(c >= '0' && c <= '9' || c >= 'a' && c <= 'f') {
			return false
		}
	}
	return true
}

func hashFile(path string) (string, error) {
	f, err := os.Open(path)
	if err != nil {
		return "", err
	}
	defer f.Close()
	h := sha256.New()
	if _, err := io.CopyBuffer(h, f, make([]byte, 4<<20)); err != nil {
		return "", err
	}
	return hex.EncodeToString(h.Sum(nil)), nil
}

// fileDigest returns a file's SHA-256. A Hugging Face LFS blob is stored
// under its SHA-256; its content is checked against that name the first
// time and trusted afterwards through the marker. A blob whose content
// does not match its name is an error: the model is not what it claims.
func fileDigest(path string) (string, error) {
	target, err := filepath.EvalSymlinks(path)
	if err != nil {
		return "", err
	}
	name := filepath.Base(target)
	if isSHA256Hex(name) && filepath.Base(filepath.Dir(target)) == "blobs" {
		if _, err := os.Stat(filepath.Join(verifiedDir(), name)); err == nil {
			return name, nil
		}
		sum, err := hashFile(target)
		if err != nil {
			return "", err
		}
		if sum != name {
			return "", fmt.Errorf("%s: content hashes to %s, not the %s it is stored as", path, sum, name)
		}
		if err := os.MkdirAll(verifiedDir(), 0o700); err == nil {
			_ = os.WriteFile(filepath.Join(verifiedDir(), name), nil, 0o600)
		}
		return name, nil
	}
	return hashFile(target)
}

// modelManifest hashes every file under dir (following the snapshot's
// symlinks) on parallel readers and returns the sorted manifest.
func modelManifest(dir string) ([]ManifestEntry, error) {
	var paths []string
	err := filepath.WalkDir(dir, func(p string, d fs.DirEntry, err error) error {
		if err != nil {
			return err
		}
		if d.IsDir() {
			if p != dir && strings.HasPrefix(d.Name(), ".") {
				return filepath.SkipDir
			}
			return nil
		}
		if strings.HasPrefix(d.Name(), ".") {
			return nil
		}
		paths = append(paths, p)
		return nil
	})
	if err != nil {
		return nil, err
	}
	if len(paths) == 0 {
		return nil, fmt.Errorf("no model files in %s", dir)
	}

	entries := make([]ManifestEntry, len(paths))
	errs := make([]error, len(paths))
	work := make(chan int)
	var wg sync.WaitGroup
	for w := 0; w < min(8, runtime.NumCPU()); w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := range work {
				rel, _ := filepath.Rel(dir, paths[i])
				info, err := os.Stat(paths[i])
				if err != nil {
					errs[i] = err
					continue
				}
				sum, err := fileDigest(paths[i])
				entries[i] = ManifestEntry{Path: filepath.ToSlash(rel), SHA256: sum, Size: info.Size()}
				errs[i] = err
			}
		}()
	}
	for i := range paths {
		work <- i
	}
	close(work)
	wg.Wait()
	for _, err := range errs {
		if err != nil {
			return nil, err
		}
	}
	sort.Slice(entries, func(a, b int) bool { return entries[a].Path < entries[b].Path })
	return entries, nil
}

// manifestDigest is the SHA-256 of the manifest in sha256sum format.
func manifestDigest(entries []ManifestEntry) string {
	h := sha256.New()
	for _, e := range entries {
		fmt.Fprintf(h, "%s  %s\n", e.SHA256, e.Path)
	}
	return hex.EncodeToString(h.Sum(nil))
}
