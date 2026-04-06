package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"

	"github.com/privasys/confidential-ai/internal/config"
	"github.com/privasys/confidential-ai/internal/handler"
)

func main() {
	cfg, err := config.Parse(os.Args[1:])
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}

	h := handler.New(cfg)
	mux := http.NewServeMux()
	h.RegisterRoutes(mux)

	logJSON("info", "confidential-ai server starting", map[string]string{
		"listen":       cfg.Listen,
		"vllm":         cfg.VLLMUpstream,
		"model":        cfg.ModelName,
		"quantization": cfg.Quantization,
		"gpu":          cfg.GPUType,
		"tee":          cfg.TeeType,
	})

	srv := &http.Server{
		Addr:         cfg.Listen,
		Handler:      mux,
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
