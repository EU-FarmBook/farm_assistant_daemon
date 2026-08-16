#!/bin/sh
set -eu

# Adapter image only. The agent runs the upstream nousresearch/hermes-agent
# image unmodified — everything specific to EU-FarmBook lives in hermes-data/
# (config template, SOUL.md, MCP bridge), which is a mounted volume, not a layer.
REGISTRY="${REGISTRY:-ghcr.io/eu-farmbook}"
IMAGE_NAME="${IMAGE_NAME:-$REGISTRY/farm_assistant_hermes}"
TAG="${1:-latest}"

echo "Building image: ${IMAGE_NAME}:${TAG}"
docker build -t "${IMAGE_NAME}:${TAG}" .

echo "Pushing image: ${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:${TAG}"

echo "Done."
echo "Image: ${IMAGE_NAME}:${TAG}"
