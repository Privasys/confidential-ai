#!/usr/bin/env bash
# Deploy confidential-ai as a standard cloud_image CONTAINER APP — the same
# path every other container app uses (no manifest, no hand-edited
# manager-apps.json). The point of this script is one thing the ad-hoc
# registration kept dropping: **per-app vault-backed /data**.
#
# A container app gets a persistent /data ONLY when created with
# `container_storage: true` (+ key_provider: enclave_generated). The deployer
# then reserves a vault key handle (apps.privasys.org/<appId>/storage-kek/v1),
# builds the LoadRequest with storage + key_handle + the fleet's vault_*
# addressing, mints the DEK in the vault, and the manager mounts a vault-backed
# LUKS volume at /data. That /data survives stop/start (DEK reconstructed from
# the constellation), so the proxy's auto-restore (/data/last-load.json) and the
# FlashInfer JIT cache persist across restarts.
#
# STANDARD: every system/fleet container that needs persistence is deployed
# this way — a cloud_image container app with container_storage:true. Do NOT
# hand-author manager-apps.json entries for storage.
#
# Usage:
#   API_BASE=https://api-test.developer.privasys.org \
#   TOKEN=<operator/admin bearer> \
#   ENCLAVE=m5-dev-ai \
#   ./deploy-app.sh
set -euo pipefail

API_BASE="${API_BASE:-https://api-test.developer.privasys.org}"
APP_NAME="${APP_NAME:-confidential-ai}"
CLOUD_IMAGE_NAME="${CLOUD_IMAGE_NAME:-confidential-ai}"
CLOUD_IMAGE_CHANNEL="${CLOUD_IMAGE_CHANNEL:-prod}"
CONTAINER_PORT="${CONTAINER_PORT:-8080}"
: "${TOKEN:?set TOKEN to an operator/admin bearer}"
: "${ENCLAVE:?set ENCLAVE to the AI fleet enclave name or id}"

auth=(-H "Authorization: Bearer ${TOKEN}")
json=(-H "Content-Type: application/json")
api() { curl -fsS "${auth[@]}" "$@"; }

echo "==> resolving enclave '${ENCLAVE}'"
ENCLAVE_ID="$(api "${API_BASE}/api/v1/enclaves" \
  | python -c "import sys,json; e=json.load(sys.stdin); m=[x for x in e if x['id']=='${ENCLAVE}' or x.get('name')=='${ENCLAVE}']; print(m[0]['id'] if m else '')")"
[ -n "${ENCLAVE_ID}" ] || { echo "enclave '${ENCLAVE}' not found"; exit 1; }
echo "    enclave_id=${ENCLAVE_ID}"

echo "==> ensuring app '${APP_NAME}' (cloud_image, container_storage=true)"
APP_ID="$(api "${API_BASE}/api/v1/apps" \
  | python -c "import sys,json; a=json.load(sys.stdin); m=[x for x in a if x['name']=='${APP_NAME}']; print(m[0]['id'] if m else '')")"
if [ -z "${APP_ID}" ]; then
  APP_ID="$(api "${json[@]}" -X POST "${API_BASE}/api/v1/apps" -d @- <<JSON | python -c "import sys,json;print(json.load(sys.stdin)['id'])"
{
  "name": "${APP_NAME}",
  "app_type": "container",
  "source_type": "cloud_image",
  "cloud_image_name": "${CLOUD_IMAGE_NAME}",
  "cloud_image_channel": "${CLOUD_IMAGE_CHANNEL}",
  "container_port": ${CONTAINER_PORT},
  "container_storage": true,
  "key_provider": "enclave_generated",
  "container_devices": ["/dev/nvidia0", "/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools"]
}
JSON
)"
  echo "    created app_id=${APP_ID}"
else
  echo "    app exists app_id=${APP_ID} (ensure its container_storage=true in the DB)"
fi

echo "==> setting minimal store listing (deploy gate)"
api "${json[@]}" -X PUT "${API_BASE}/api/v1/apps/${APP_ID}/store" -d @- >/dev/null <<'JSON'
{ "store_description": "Privasys Confidential AI inference proxy.", "store_category": "Developer Tools" }
JSON

echo "==> finding ready version"
VERSION_ID=""
for _ in $(seq 1 30); do
  VERSION_ID="$(api "${API_BASE}/api/v1/apps/${APP_ID}/versions" \
    | python -c "import sys,json; v=json.load(sys.stdin); r=[x for x in v if x['status'] in ('ready','built')]; print(r[0]['id'] if r else '')")"
  [ -n "${VERSION_ID}" ] && break
  sleep 5
done
[ -n "${VERSION_ID}" ] || { echo "no ready version"; exit 1; }
echo "    version_id=${VERSION_ID}"

echo "==> deploying to ${ENCLAVE} (vault-backed /data provisioned by the deployer)"
api "${json[@]}" -X POST \
  "${API_BASE}/api/v1/apps/${APP_ID}/versions/${VERSION_ID}/deploy" \
  -d "{\"enclave_id\": \"${ENCLAVE_ID}\"}" >/dev/null

echo "==> waiting for active"
for _ in $(seq 1 60); do
  st="$(api "${API_BASE}/api/v1/apps/${APP_ID}/deployments" \
    | python -c "import sys,json; d=json.load(sys.stdin); print(d[0]['status'] if d else '')" 2>/dev/null || true)"
  echo "    status=${st}"
  [ "${st}" = "active" ] && { echo "DONE: confidential-ai deployed with vault-backed /data"; exit 0; }
  [ "${st}" = "failed" ] && { echo "deploy failed"; exit 1; }
  sleep 5
done
echo "did not reach active in time"; exit 1
