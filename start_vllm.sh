#!/bin/bash
export HF_ENDPOINT=https://hf-mirror.com
/home/tiantianyi/code/federated-mcts/.venv/bin/vllm serve Qwen/Qwen2.5-32B-Instruct   --tensor-parallel-size 2 --max-model-len 16384 --port 8000
