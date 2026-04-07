# Contributing to Confidential AI

Thank you for your interest in contributing to Confidential AI.

## Getting Started

1. Fork and clone the repository
2. Install [Go 1.22+](https://go.dev/dl/)
3. Build: `go build -o confidential-ai ./cmd/server/`
4. Run tests: `go test ./...`

## Project Structure

| Path | Description |
|------|-------------|
| `cmd/server/` | CLI entrypoint and flag parsing |
| `internal/proxy/` | Reverse proxy to vLLM backend |
| `internal/reproducibility/` | Reproducibility metadata injection |
| `internal/health/` | Health endpoint handler |
| `internal/config/` | Configuration loading |
| `deploy/` | Deployment scripts and Docker entrypoint |
| `privasys.json` | MCP tool manifest for the developer platform |

## Making Changes

- Follow standard Go conventions (`gofmt`, `go vet`)
- Use [Conventional Commits](https://www.conventionalcommits.org/) for commit messages
- All commits must be GPG-signed
- Add tests for new functionality

## Submitting a Pull Request

1. Create a feature branch from `main`
2. Make your changes with clear, focused commits
3. Ensure `go test ./...` passes
4. Open a PR against `main` with a description of the change

## Reporting Issues

Please use [GitHub Issues](https://github.com/Privasys/confidential-ai/issues) to report bugs or request features.

## License

By contributing, you agree that your contributions will be licensed under the [GNU Affero General Public License v3.0](LICENSE).
