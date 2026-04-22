#!/bin/bash
# Deploy a GCP Confidential VM with TDX + H100 for confidential AI inference.
# Uses Enclave OS Virtual GPU image with pre-installed NVIDIA CC-patched drivers.
# Requires: gcloud CLI authenticated, quota for a3-highgpu-1g.
# Usage: ./deploy-gcp.sh [instance-name] [zone]
set -euo pipefail

NAME="${1:-confidential-ai-test}"
ZONE="${2:-europe-west4-c}"
PROJECT=$(gcloud config get-value project)
MACHINE_TYPE="a3-highgpu-1g"
IMAGE="enclave-os-virtual-gpu-v0-5-0"  # built by enclave-os-virtual CI

echo "Deploying $NAME in $ZONE ($PROJECT)"
echo "  Machine type: $MACHINE_TYPE (1x H100 80GB, Intel TDX)"
echo "  Image:        $IMAGE (Enclave OS Virtual GPU)"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --confidential-compute-type=TDX \
  --provisioning-model=SPOT \
  --maintenance-policy=TERMINATE \
  --network-interface=nic-type=GVNIC \
  --create-disk=auto-delete=yes,boot=yes,image=projects/${PROJECT}/global/images/${IMAGE},size=40,type=pd-balanced \
  --shielded-vtpm \
  --shielded-integrity-monitoring \
  --metadata=enable-oslogin=FALSE

echo ""
echo "Instance $NAME created."
echo ""
echo "The VM uses Enclave OS Virtual GPU with read-only erofs root."
echo "NVIDIA CC-patched drivers load via the GCP startup script."
echo ""
echo "SSH:  gcloud compute ssh $NAME --zone=$ZONE"
echo "Stop: gcloud compute instances stop $NAME --zone=$ZONE --discard-local-ssd=false"
