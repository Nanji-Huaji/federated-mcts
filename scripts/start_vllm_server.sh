#!/bin/bash
# Start a vLLM server for local model serving.
# Models are auto-downloaded from HuggingFace if not cached.
#
# Usage:
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-7B-Instruct
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-7B-Instruct --port 8080
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-7B-Instruct --hf-mirror
#   bash scripts/start_vllm_server.sh Qwen/Qwen2.5-7B-Instruct --download-only

set -euo pipefail

MODEL="${1:?Usage: $0 <model_name> [options]}"
shift

PORT=8000
GPU_MEM_UTIL=0.9
MAX_MODEL_LEN=4096
HF_MIRROR=""
DOWNLOAD_ONLY=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        --port) PORT="$2"; shift 2 ;;
        --gpu-memory-utilization) GPU_MEM_UTIL="$2"; shift 2 ;;
        --max-model-len) MAX_MODEL_LEN="$2"; shift 2 ;;
        --hf-mirror) HF_MIRROR="https://hf-mirror.com"; shift ;;
        --download-only) DOWNLOAD_ONLY=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

if [ -n "$HF_MIRROR" ]; then
    export HF_ENDPOINT="$HF_MIRROR"
    echo "Using HF mirror: $HF_ENDPOINT"
fi

if [ "$DOWNLOAD_ONLY" = true ]; then
    echo "Downloading model (no server start)..."
    echo "  Model: $MODEL"
    python3 -c "
from huggingface_hub import snapshot_download
snapshot_download('$MODEL')
print('Download complete.')
"
    exit 0
fi

echo "Starting vLLM server..."
echo "  Model:            $MODEL"
echo "  Port:             $PORT"
echo "  GPU mem util:     $GPU_MEM_UTIL"
echo "  Max model len:    $MAX_MODEL_LEN"
echo ""

vllm serve "$MODEL" \
    --host 0.0.0.0 \
    --port "$PORT" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --max-model-len "$MAX_MODEL_LEN"

echo "vLLM server stopped."
