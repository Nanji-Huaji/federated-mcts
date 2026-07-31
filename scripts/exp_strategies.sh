#!/bin/bash
# Strategy ablation experiment — compare round_robin vs difficulty vs bandit
# Usage: bash scripts/exp_strategies.sh

set -euo pipefail

TASK=game24
START=900
END=920
MODEL_CONFIG=configs/model_config.json

COMMON="--task $TASK --task_start_index $START --task_end_index $END \
  --model_config $MODEL_CONFIG \
  --method_generate propose --method_evaluate value --method_select greedy \
  --n_evaluate_sample 1 --n_select_sample 5 --temperature 0.7 \
  --solve_method federated_solve"

echo "=== Experiment: Strategy Ablation ==="
echo "Task: $TASK | Range: $START-$END ($((END - START)) tasks)"
echo ""

for strategy in round_robin difficulty bandit; do
    echo "--- Running: $strategy ---"
    python scripts/merged_run.py $COMMON --assign_strategy "$strategy"
    echo ""
done

echo "=== Done ==="
echo "Results in: logs/$TASK/federated_solve/"
echo "Compare the _performance.json files for acc, cost, latency"
