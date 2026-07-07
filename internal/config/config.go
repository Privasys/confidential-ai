package config

import (
	"flag"
	"fmt"
	"os"
	"strings"
	"time"
)

// Config holds all server configuration.
type Config struct {
	Listen       string // HTTP listen address
	VLLMUpstream string // vLLM backend URL
	ModelsDir    string // directory containing model subdirectories (e.g. /models)
	VLLMPort     int    // port for vLLM subprocess (default 8000)
	ModelName    string // model identifier for metadata (legacy, dynamic in new mode)
	ModelDigest  string // SHA-256 of model weights index (legacy, dynamic in new mode)
	Quantization string // quantization method (awq, gptq, fp8, etc.)
	GPUType      string // GPU hardware identifier
	ImageDigest  string // CVM image SHA256 digest
	CUDAVersion  string // CUDA version string
	VLLMVersion  string // vLLM version string
	TeeType      string // TEE type: tdx, sev-snp

	// RoothashDir is the directory written by disk-mounter on the host
	// containing per-model dm-verity root hashes (one file per model,
	// named <model>.roothash). When a load request resolves to a model
	// whose roothash file is present, the proxy publishes that root
	// hash as the OID 3.5 (MODEL_DIGEST) attestation extension instead
	// of hashing the safetensors index.
	RoothashDir string

	// StateFile is the path on the per-container encrypted volume where
	// the most recently successful /v1/models/load request is
	// persisted. On container start the proxy reads this file and
	// auto-issues the same Load so a restart (typically after a Spot-VM
	// reboot) recovers the previously-served model without manual
	// orchestration. Empty disables the feature.
	StateFile string

	// LoadToken, when non-empty, is accepted as a Bearer credential on
	// POST /v1/models/load and POST /v1/models/unload. It is now a LEGACY
	// FALLBACK for the direct CLI/owner path: the primary gate is the
	// OIDC manager role (see OIDCIssuer/ManagerRole). When both LoadToken
	// and OIDCIssuer are empty the endpoints remain open (dev mode).
	LoadToken string

	// OIDCIssuer is the platform OIDC issuer whose JWKS validates bearer
	// tokens on privileged endpoints. When non-empty, /v1/models/{load,
	// unload} require a token from this issuer carrying ManagerRole (the
	// management-service service account presents exactly such a token).
	// This mirrors the enclave manager's own auth model. env: OIDC_ISSUER.
	OIDCIssuer string

	// OIDCAudience, when non-empty, is the required `aud` on a verified
	// token. Empty skips the audience check. env: OIDC_AUDIENCE.
	OIDCAudience string

	// ManagerRole is the role a token must carry (in the `roles` claim)
	// to load/unload models. Default: privasys-platform:manager.
	// env: MANAGER_ROLE.
	ManagerRole string

	// RevokedSidsURL is the IdP feed of revoked session ids that the proxy
	// polls so a revoked API key (a token whose sid was revoked) is rejected
	// without a per-request callout. Empty derives it from OIDCIssuer
	// (<issuer>/sessions/revoked). env: REVOKED_SIDS_URL.
	RevokedSidsURL string

	// RevokedSidsInterval is the revoked-sid poll cadence. Default 60s.
	// env: REVOKED_SIDS_INTERVAL.
	RevokedSidsInterval time.Duration

	// MCPRATLS routes the agent loop's MCP calls (tool discovery + tool
	// invocations, transport privasys_http) over per-request attested
	// RA-TLS connections to the tool enclaves instead of gateway-terminated
	// HTTPS. The enclave gateways refuse plaintext app traffic on the
	// terminated leg (sealed-transport-required), so this is REQUIRED for
	// tools to work on the platform; disable only for local dev against
	// plain-HTTP MCP servers. Default true. env: MCP_RATLS.
	MCPRATLS bool

	// MCPServers, when non-empty, enables the agentic tool-call loop
	// on POST /v1/chat/completions. Format (env MCP_SERVERS):
	//
	//   name1=https://url1[?bearer=1],name2=https://url2,...
	//
	// `bearer=1` opts the server in to receiving the user's
	// Authorization header from the original chat request (required
	// for private-rag, optional for stateless tools like lightpanda).
	// When MCPServers is empty the proxy behaves exactly as before
	// (pure pass-through to vLLM).
	MCPServers string

	// ToolSpecURL, when non-empty, enables the background tool-spec
	// puller. The proxy polls this URL every ToolSpecInterval and
	// atomically replaces the agent.Catalog's server list with the
	// returned spec string (see internal/agent/spec.go::ParseServerSpec
	// for format). This is how managed instances (e.g. confidential-ai
	// running inside an enclave alongside the workload manager) pick
	// up fleet-level tool-set changes without a container restart.
	//
	// The endpoint must return JSON: {"spec":"...","generation":"..."}.
	// When ToolSpecURL is empty the puller is disabled and the tool
	// catalogue is fixed to whatever MCPServers resolved to at startup.
	ToolSpecURL string

	// ToolSpecToken, when non-empty, is sent as `Authorization: Bearer
	// <token>` on every ToolSpecURL poll. Typically a static
	// machine-to-machine credential issued to the enclave by the
	// management service.
	ToolSpecToken string

	// ToolSpecInterval is the polling cadence for ToolSpecURL.
	// Defaults to 60s when zero. Ignored when ToolSpecURL is empty.
	ToolSpecInterval time.Duration

	// ToolGrantJWKSURL, when non-empty, enables per-request user tool
	// grants. The proxy verifies the X-Privasys-Tool-Grant header (an
	// ES256 JWS minted by the chat back-end) against this JWKS and, for a
	// valid grant, unions the grant's tool servers with the configured
	// catalogue for that single request. The browser supplies only the
	// grant, never a raw server URL. Empty disables the feature (the
	// header is ignored and only the configured catalogue is used).
	ToolGrantJWKSURL string

	// ToolGrantAudience is the `aud` a grant must carry to be accepted —
	// this instance's id. Empty skips the audience check (dev only).
	ToolGrantAudience string

	// CORSOrigins is a comma-separated allowlist of HTTP Origins that
	// receive Access-Control-Allow-* response headers. Defaults to the
	// Privasys chat front-ends. Empty disables CORS entirely (browser
	// requests from any origin will be blocked by the SOP).
	CORSOrigins string

	// --- Billing (the pricing model) ---
	//
	// The proxy meters every completed inference by the vLLM-reported
	// prompt/completion token counts and pushes them, per request, to
	// the management-service AI-usage endpoint (UsageReportURL). That
	// service — not this enclave — holds the credit-ledger credential
	// and performs the priced debit, then returns the account's freeze
	// state so the proxy can refuse inference at zero balance. The
	// enclave is deliberately given a usage-only machine credential
	// (UsageReportToken), never the ledger's grant-capable token.
	//
	// All four fields may be supplied by env/flags, but container apps
	// receive them at runtime via POST /configure (the configure-then-
	// freeze pattern — the container load envelope carries no env vars).
	// The delivered config is persisted to BillingConfigFile and reloaded
	// on restart. When UsageReportURL or BillingAccountID is empty,
	// metering and the balance gate are disabled and the proxy serves
	// unmetered.

	// BillingAccountID is the deployment-owner account charged for
	// inference (per-caller attribution is a later phase). env:
	// BILLING_ACCOUNT_ID.
	BillingAccountID string

	// UsageReportURL is the management-service endpoint the proxy POSTs
	// token usage to (e.g. https://manager.internal/api/v1/enclave/ai-usage).
	// env: USAGE_REPORT_URL.
	UsageReportURL string

	// UsageReportToken is the EnclaveToken bearer presented to
	// UsageReportURL. env: USAGE_REPORT_TOKEN.
	UsageReportToken string

	// BillingModel overrides the model slug reported to the usage
	// endpoint for price-book lookup. Defaults to ModelName / the
	// dynamically loaded model name when empty. env: BILLING_MODEL.
	BillingModel string

	// BillingConfigFile is the path on the per-container encrypted volume
	// where billing configuration delivered via POST /configure is
	// persisted. On container start the proxy reloads this file so
	// metering survives a restart (the manager re-freezes after every
	// restart and the orchestrator re-delivers via /configure, but the
	// persisted copy lets the proxy restore billing without waiting).
	// Empty disables persistence. env: BILLING_CONFIG_FILE.
	BillingConfigFile string
}

// Parse reads configuration from flags and environment, returning it.
func Parse(args []string) (*Config, error) {
	fs := flag.NewFlagSet("confidential-ai", flag.ContinueOnError)

	cfg := &Config{}

	fs.StringVar(&cfg.Listen, "listen", envOr("LISTEN_ADDR", defaultListenAddr()),
		"HTTP listen address (env: LISTEN_ADDR; defaults to :$PORT when the platform allocates one)")
	fs.StringVar(&cfg.VLLMUpstream, "vllm-upstream", envOr("VLLM_UPSTREAM", "http://localhost:8000"),
		"vLLM backend URL (env: VLLM_UPSTREAM)")
	fs.StringVar(&cfg.ModelsDir, "models-dir", envOr("MODELS_DIR", "/models"),
		"Directory containing model subdirectories (env: MODELS_DIR)")
	fs.IntVar(&cfg.VLLMPort, "vllm-port", 8000,
		"Port for vLLM subprocess to listen on")
	fs.StringVar(&cfg.ModelName, "model", envOr("MODEL_NAME", ""),
		"Model name for reproducibility metadata (env: MODEL_NAME)")
	fs.StringVar(&cfg.ModelDigest, "model-digest", envOr("MODEL_DIGEST", ""),
		"SHA-256 of model weights index for attestation (env: MODEL_DIGEST)")
	fs.StringVar(&cfg.Quantization, "quantization", envOr("QUANTIZATION", ""),
		"Quantization method (env: QUANTIZATION)")
	fs.StringVar(&cfg.GPUType, "gpu-type", envOr("GPU_TYPE", "H100-80GB"),
		"GPU hardware type (env: GPU_TYPE)")
	fs.StringVar(&cfg.ImageDigest, "image-digest", envOr("IMAGE_DIGEST", ""),
		"CVM image SHA256 digest (env: IMAGE_DIGEST)")
	fs.StringVar(&cfg.CUDAVersion, "cuda-version", envOr("CUDA_VERSION", "12.6.3"),
		"CUDA version (env: CUDA_VERSION)")
	fs.StringVar(&cfg.VLLMVersion, "vllm-version", envOr("VLLM_VERSION", "0.22.1"),
		"vLLM version (env: VLLM_VERSION)")
	fs.StringVar(&cfg.TeeType, "tee-type", envOr("TEE_TYPE", "tdx"),
		"TEE type: tdx or sev-snp (env: TEE_TYPE)")
	fs.StringVar(&cfg.RoothashDir, "roothash-dir", envOr("ROOTHASH_DIR", "/var/lib/enclave-os/model-roothashes"),
		"Directory of per-model dm-verity root hashes (env: ROOTHASH_DIR)")
	fs.StringVar(&cfg.StateFile, "state-file", envOr("STATE_FILE", "/data/last-load.json"),
		"Path where the last successful Load request is persisted for auto-restore on restart (env: STATE_FILE; empty disables)")
	fs.StringVar(&cfg.LoadToken, "load-token", envOr("LOAD_TOKEN", ""),
		"Legacy fallback bearer accepted on /v1/models/{load,unload} alongside the OIDC manager role (env: LOAD_TOKEN)")
	fs.StringVar(&cfg.OIDCIssuer, "oidc-issuer", envOr("OIDC_ISSUER", "https://privasys.id"),
		"Platform OIDC issuer whose JWKS validates end-user (inference) and manager (load/unload) bearer tokens. Inference authentication is mandatory, so an empty issuer leaves no way to authenticate callers and rejects all inference with 401 (env: OIDC_ISSUER)")
	fs.StringVar(&cfg.OIDCAudience, "oidc-audience", envOr("OIDC_AUDIENCE", ""),
		"Required aud on a verified token; empty skips the audience check (env: OIDC_AUDIENCE)")
	fs.StringVar(&cfg.ManagerRole, "manager-role", envOr("MANAGER_ROLE", "privasys-platform:manager"),
		"Role required on /v1/models/{load,unload} tokens (env: MANAGER_ROLE)")
	fs.StringVar(&cfg.RevokedSidsURL, "revoked-sids-url", envOr("REVOKED_SIDS_URL", ""),
		"IdP revoked-session feed to poll; empty derives <OIDC_ISSUER>/sessions/revoked (env: REVOKED_SIDS_URL)")
	fs.DurationVar(&cfg.RevokedSidsInterval, "revoked-sids-interval", envDuration("REVOKED_SIDS_INTERVAL", 60*time.Second),
		"Revoked-sid poll cadence (env: REVOKED_SIDS_INTERVAL)")
	fs.BoolVar(&cfg.MCPRATLS, "mcp-ratls", envBool("MCP_RATLS", true),
		"Carry MCP tool calls over per-request attested RA-TLS to the tool enclaves; disable only for local dev against plain-HTTP servers (env: MCP_RATLS)")
	fs.StringVar(&cfg.MCPServers, "mcp-servers", envOr("MCP_SERVERS", ""),
		"Comma-separated <name>=<url>[?bearer=1] list of MCP servers to expose as tools (env: MCP_SERVERS)")
	fs.StringVar(&cfg.ToolSpecURL, "tool-spec-url", envOr("TOOL_SPEC_URL", ""),
		"When set, the proxy polls this URL for {spec,generation} and hot-reloads the tool catalogue (env: TOOL_SPEC_URL)")
	fs.StringVar(&cfg.ToolSpecToken, "tool-spec-token", envOr("TOOL_SPEC_TOKEN", ""),
		"Bearer token sent on every tool-spec-url poll (env: TOOL_SPEC_TOKEN)")
	fs.DurationVar(&cfg.ToolSpecInterval, "tool-spec-interval", envDuration("TOOL_SPEC_INTERVAL", 60*time.Second),
		"How often to poll tool-spec-url (env: TOOL_SPEC_INTERVAL, e.g. 30s)")
	fs.StringVar(&cfg.ToolGrantJWKSURL, "tool-grant-jwks-url", envOr("TOOL_GRANT_JWKS_URL", ""),
		"When set, verify X-Privasys-Tool-Grant against this JWKS and union the grant's tools per request (env: TOOL_GRANT_JWKS_URL)")
	fs.StringVar(&cfg.ToolGrantAudience, "tool-grant-audience", envOr("TOOL_GRANT_AUDIENCE", ""),
		"Expected aud claim on a tool-grant (this instance's id); empty skips the check (env: TOOL_GRANT_AUDIENCE)")
	fs.StringVar(&cfg.CORSOrigins, "cors-origins", envOr("CORS_ORIGINS", "https://chat.privasys.org,https://chat-test.privasys.org,http://localhost:4210,http://localhost:3000"),
		"Comma-separated CORS Origin allowlist (env: CORS_ORIGINS)")

	fs.StringVar(&cfg.BillingAccountID, "billing-account-id", envOr("BILLING_ACCOUNT_ID", ""),
		"Deployment-owner account charged for inference; empty disables metering (env: BILLING_ACCOUNT_ID)")
	fs.StringVar(&cfg.UsageReportURL, "usage-report-url", envOr("USAGE_REPORT_URL", ""),
		"management-service AI-usage endpoint the proxy reports token usage to; empty disables metering (env: USAGE_REPORT_URL)")
	fs.StringVar(&cfg.UsageReportToken, "usage-report-token", envOr("USAGE_REPORT_TOKEN", ""),
		"EnclaveToken bearer presented to usage-report-url (env: USAGE_REPORT_TOKEN)")
	fs.StringVar(&cfg.BillingModel, "billing-model", envOr("BILLING_MODEL", ""),
		"Model slug reported for price-book lookup; defaults to the served model name (env: BILLING_MODEL)")
	fs.StringVar(&cfg.BillingConfigFile, "billing-config-file", envOr("BILLING_CONFIG_FILE", "/data/billing-config.json"),
		"Path where billing config from POST /configure is persisted and reloaded on restart; empty disables (env: BILLING_CONFIG_FILE)")

	if err := fs.Parse(args); err != nil {
		return nil, err
	}
	if cfg.Listen == "" {
		return nil, fmt.Errorf("listen address is required: set --listen, LISTEN_ADDR, or the platform-injected PORT")
	}
	return cfg, nil
}

// defaultListenAddr honours the platform-allocated $PORT (host networking
// makes a container's listen port its host port, so the management-service
// assigns it and injects PORT). Returns "" when neither PORT nor an explicit
// --listen/LISTEN_ADDR is set — there is no hard-coded fallback port; the
// caller rejects an empty listen address.
func defaultListenAddr() string {
	if p := os.Getenv("PORT"); p != "" {
		return ":" + p
	}
	return ""
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}

func envBool(key string, fallback bool) bool {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	switch strings.ToLower(strings.TrimSpace(v)) {
	case "1", "true", "yes", "on":
		return true
	case "0", "false", "no", "off":
		return false
	default:
		return fallback
	}
}

func envDuration(key string, fallback time.Duration) time.Duration {
	v := os.Getenv(key)
	if v == "" {
		return fallback
	}
	d, err := time.ParseDuration(v)
	if err != nil {
		return fallback
	}
	return d
}
