#!/bin/bash
# Download the ONNX models and pipeline config the mod uses (pinned Hugging Face revision, ~2.3 GB).
set -euo pipefail
DIR="${1:-$(cd "$(dirname "$0")/.." && pwd)/models}"
REV=ad2df557eca5645f588766101cf3bc3682455c3e
mkdir -p "$DIR"
for f in coarse_model.onnx decoder_model.onnx pipeline_data.json world_pipeline_config.json base_model.onnx; do
  [ -s "$DIR/$f" ] || curl -fL --progress-bar -o "$DIR/$f" \
    "https://huggingface.co/xandergos/terrain-diffusion-30m-onnx/resolve/$REV/$f?download=true"
done
ls -la "$DIR"
