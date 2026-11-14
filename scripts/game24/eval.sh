#!/usr/bin/env bash
set -euo pipefail

PYTHONPATH=src .venv/bin/python -m federated_mcts \
    --task game24 \
    --task_start_index 200 \
    --task_end_index 210 \
    --model_config model_config.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method tot \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    "$@"
