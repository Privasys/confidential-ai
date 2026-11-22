// Copyright (c) Privasys. All rights reserved.
// Licensed under the GNU Affero General Public License v3.0.

package models

import (
	"context"
	"crypto/rand"
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"log"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
)

// Instance watermarking.
//
// vLLM (0.34+) can bias sampling with a keyed Gumbel watermark, and the same
// key later tells whether a text came from this instance (a p-value). The key
// is the instance's: generated here, inside the enclave, the first time a
// load asks for watermarking, kept on the encrypted app volume, and never
// returned by any route or written to any log. Nobody outside the enclave,
// the operator included, can mark text as this instance's or test text
// without going through the attested detection route.
//
// Watermarking is opt-in per request: the proxy sends watermarking=false
// for a request that does not set the field (vLLM's own default is on), so
// a caller asks for the mark explicitly.

// watermarkDetectorPort is the loopback port of the detection helper.
// 8000-8002 are the vLLM instances; 8080 is reserved.
const watermarkDetectorPort = 8009

// watermarkKeyPath is where the instance key lives, beside the model cache
// on the encrypted volume.
func watermarkKeyPath() string {
	return filepath.Join(vllmHome(), "privasys-watermark", "key")
}

// instanceWatermarkKey returns the instance key, creating it on first use.
// The key is 64 bits, the width vLLM's Philox PRF takes.
func instanceWatermarkKey() (uint64, error) {
	path := watermarkKeyPath()
	if b, err := os.ReadFile(path); err == nil {
		return strconv.ParseUint(strings.TrimSpace(string(b)), 10, 64)
	} else if !errors.Is(err, os.ErrNotExist) {
		return 0, err
	}
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return 0, err
	}
	var b [8]byte
	if _, err := rand.Read(b[:]); err != nil {
		return 0, err
	}
	key := binary.BigEndian.Uint64(b[:])
	f, err := os.OpenFile(path, os.O_CREATE|os.O_EXCL|os.O_WRONLY, 0o600)
	if errors.Is(err, os.ErrExist) {
		return instanceWatermarkKey() // another load won the race: use its key
	}
	if err != nil {
		return 0, err
	}
	if _, err := f.WriteString(strconv.FormatUint(key, 10)); err != nil {
		f.Close()
		return 0, err
	}
	if err := f.Close(); err != nil {
		return 0, err
	}
	log.Printf("[watermark] instance key created (id %s)", watermarkKeyID(key))
	return key, nil
}

// watermarkKeyID names a key without revealing it, so a reproducibility
// record or a detection result can say which key it refers to.
func watermarkKeyID(key uint64) string {
	h := sha256.New()
	h.Write([]byte("privasys-watermark-key-id\x00"))
	var b [8]byte
	binary.BigEndian.PutUint64(b[:], key)
	h.Write(b[:])
	return hex.EncodeToString(h.Sum(nil))[:16]
}

// watermarkConfigArg is the --watermark-config value. It carries the key, so
// it is never logged; vLLM's own startup log echoes non-default arguments,
// which is why every engine output line passes through redactWatermarkKey.
func watermarkConfigArg(key uint64) string {
	return fmt.Sprintf(`{"algorithm":"gumbel","key":%d}`, key)
}

// redactWatermarkKey removes the key from a line of engine output.
func redactWatermarkKey(line, keyStr string) string {
	if keyStr == "" {
		return line
	}
	return strings.ReplaceAll(line, keyStr, "[watermark-key]")
}

// startWatermarkDetector runs the detection helper for the loaded model: a
// small CPU-only HTTP server on loopback that tokenizes a text with the
// model's tokenizer and scores it with vLLM's Gumbel detector and the
// instance key (read from the key file, never passed on a command line).
// It lives as long as ctx, the load's context.
func startWatermarkDetector(ctx context.Context, tokenizerDir string) error {
	cmd := exec.CommandContext(ctx, "python3", "/opt/confidential-ai/watermark_detect.py",
		"--tokenizer", tokenizerDir,
		"--key-file", watermarkKeyPath(),
		"--port", strconv.Itoa(watermarkDetectorPort))
	cmd.Stdout = os.Stderr
	cmd.Stderr = os.Stderr
	cmd.SysProcAttr = &syscall.SysProcAttr{Setpgid: true}
	if err := cmd.Start(); err != nil {
		return err
	}
	go func() {
		if err := cmd.Wait(); err != nil && ctx.Err() == nil {
			log.Printf("[watermark] detector exited: %v", err)
		}
	}()
	return nil
}

// WatermarkDetectorURL is the detection helper's base URL.
func WatermarkDetectorURL() string {
	return fmt.Sprintf("http://127.0.0.1:%d", watermarkDetectorPort)
}
