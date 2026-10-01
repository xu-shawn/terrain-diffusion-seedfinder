#!/bin/bash
# One-time setup on a rented CUDA box (tested on nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04).
# Run after `scripts/hpush.sh ip:port` has copied the code to /root/td.
set -euo pipefail
export DEBIAN_FRONTEND=noninteractive
apt-get update -qq && apt-get install -y -qq python3-pip rsync curl > /dev/null
pip3 install -q onnxruntime-gpu==1.20.0 numpy numba scipy pyfastnoiselite==0.0.6 pillow
mkdir -p /root/td/work
bash /root/td/scripts/fetch_models.sh /root/td/models
python3 -c "import onnxruntime as ort; print(ort.__version__, ort.get_available_providers())"
nvidia-smi --query-gpu=name,memory.total --format=csv
