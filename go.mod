module github.com/privasys/confidential-ai

go 1.22

require github.com/google/uuid v1.6.0

require enclave-os-mini/clients/go v0.0.0-00010101000000-000000000000

replace enclave-os-mini/clients/go => ../ra-tls-clients/go
