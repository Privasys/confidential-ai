#!/bin/bash
# Deploy a GCP Confidential VM with TDX + H100 for testing.
# Requires: gcloud CLI authenticated, quota for a3-highgpu-1g.
# Usage: ./deploy-gcp.sh [instance-name] [zone]
set -euo pipefail

NAME="${1:-confidential-ai-test}"
ZONE="${2:-europe-west4-a}"
PROJECT=$(gcloud config get-value project)
MACHINE_TYPE="a3-highgpu-1g"

echo "Deploying $NAME in $ZONE ($PROJECT)"
echo "  Machine type: $MACHINE_TYPE (1x H100 80GB, Intel TDX)"

gcloud compute instances create "$NAME" \
  --project="$PROJECT" \
  --zone="$ZONE" \
  --machine-type="$MACHINE_TYPE" \
  --confidential-compute-type=TDX \
  --provisioning-model=SPOT \
  --image-family=ubuntu-2204-lts \
  --image-project=ubuntu-os-cloud \
  --boot-disk-size=200GB \
  --boot-disk-type=pd-ssd \
  --scopes=cloud-platform \
  --metadata=startup-script='#!/bin/bash
set -ex

# Install NVIDIA driver
apt-get update
apt-get install -y nvidia-driver-550 nvidia-utils-550

# Install NVIDIA Container Toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
echo "deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://nvidia.github.io/libnvidia-container/stable/deb/amd64 /" > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update
apt-get install -y nvidia-container-toolkit

# Install Docker
curl -fsSL https://get.docker.com | sh
nvidia-ctk runtime configure --runtime=docker
systemctl restart docker

echo "Setup complete. Run: docker run --gpus all confidential-ai"
'

echo ""
echo "Instance $NAME created. Wait for startup script to complete (~5 min)."
echo "SSH: gcloud compute ssh $NAME --zone=$ZONE"
echo "Stop: gcloud compute instances stop $NAME --zone=$ZONE"
