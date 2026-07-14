package main

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/privasys/confidential-ai/internal/agent/specsync"
	"github.com/privasys/confidential-ai/internal/config"
	"github.com/privasys/confidential-ai/internal/handler"
	"github.com/privasys/confidential-ai/internal/models"
)

func main() {
	cfg, err := config.Parse(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	// Create the model fleet for dynamic model loading (generate + embed +
	// rerank instances, one vLLM subprocess each on vllmPort+0/+1/+2).
	// If --models-dir points to an existing directory, enable dynamic mode.
	// Otherwise fall back to legacy mode (vLLM started by entrypoint.sh).
	var fleet *models.Fleet
	if info, err := os.Stat(cfg.ModelsDir); err == nil && info.IsDir() {
		fleet = models.NewFleet(cfg.ModelsDir, cfg.VLLMPort, cfg.RoothashDir, cfg.StateFile)
		logJSON("info", "dynamic model loading enabled", map[string]string{
			"models_dir": cfg.ModelsDir,
			"state_file": cfg.StateFile,
		})
		if restored, err := fleet.RestoreFromDisk(); err != nil {
			logJSON("warn", "model auto-restore failed", map[string]string{"error": err.Error()})
		} else if len(restored) > 0 {
			fields := map[string]string{}
			for task, model := range restored {
				fields[string(task)] = model
			}
			logJSON("info", "model auto-restore initiated", fields)
		}
	} else {
		logJSON("info", "legacy mode (vLLM managed by entrypoint)", nil)
	}

	h := handler.New(cfg, fleet)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	// Tool-spec puller: when --tool-spec-url is set, poll it and hot-
	// reload the agent catalogue. The puller owns its own goroutine
	// cancelled by ctx on shutdown. Disabled when AgentCatalog() is
	// nil (no MCP_SERVERS and no --tool-spec-url).
	ctx, stop := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer stop()

	// Billing: start the background inference-metering loop (no-op when
	// metering is not configured), then restore any billing config
	// persisted on the encrypted volume by a previous POST /configure so
	// metering survives a restart.
	h.StartBilling(ctx)
	h.RestorePersistedBilling()

	if cfg.ToolSpecURL != "" {
		if cat := h.AgentCatalog(); cat != nil {
			s := specsync.New(cfg.ToolSpecURL, cfg.ToolSpecToken, cfg.ToolSpecInterval, nil, cat).
				OnGrant(h.SetGrantVerifierFromSpec)
			go s.Run(ctx)
			logJSON("info", "tool-spec puller started", map[string]string{
				"url":      cfg.ToolSpecURL,
				"interval": cfg.ToolSpecInterval.String(),
			})
		} else {
			logJSON("warn", "tool-spec-url set but agent catalogue is nil; puller skipped", nil)
		}
	}

	logJSON("info", "confidential-ai server starting", map[string]string{
		"listen":       cfg.Listen,
		"vllm":         cfg.VLLMUpstream,
		"model":        cfg.ModelName,
		"quantization": cfg.Quantization,
		"gpu":          cfg.GPUType,
		"tee":          cfg.TeeType,
		"models_dir":   cfg.ModelsDir,
	})

	srv := &http.Server{
		Addr:         cfg.Listen,
		Handler:      handler.CORS(cfg.CORSOrigins, mux),
		ReadTimeout:  30 * time.Second,
		WriteTimeout: 10 * time.Minute, // LLM inference can be slow
		IdleTimeout:  120 * time.Second,
	}

	if err := srv.ListenAndServe(); err != nil {
		logJSON("fatal", "server exited", map[string]string{"error": err.Error()})
		os.Exit(1)
	}
}

func logJSON(level, msg string, fields map[string]string) {
	entry := map[string]string{
		"level": level,
		"msg":   msg,
		"time":  time.Now().UTC().Format(time.RFC3339),
	}
	for k, v := range fields {
		entry[k] = v
	}
	json.NewEncoder(os.Stderr).Encode(entry)
}
