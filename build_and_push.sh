#!/bin/sh
set -eu

# Adapter image only. The agent runs the upstream nousresearch/hermes-agent
# image unmodified — everything specific to EU-FarmBook lives in hermes-data/
# (config template, SOUL.md, MCP bridge), which is a mounted volume, not a layer.
REGISTRY="${REGISTRY:-ghcr.io/eu-farmbook}"
IMAGE_NAME="${IMAGE_NAME:-$REGISTRY/farm_assistant_hermes}"
TAG="${1:-latest}"

echo "Building image: ${IMAGE_NAME}:${TAG}"
# --network=host for the same reason farm_assistant_um needs it: the default
# bridge network on this host cannot resolve public DNS, so pip install fails
# with "No matching distribution found" for packages that plainly exist. It
# only bites when the dependency layer is not cached.
docker build --network=host -t "${IMAGE_NAME}:${TAG}" .

echo "Pushing image: ${IMAGE_NAME}:${TAG}"
docker push "${IMAGE_NAME}:${TAG}"

echo "Done."
echo "Image: ${IMAGE_NAME}:${TAG}"
