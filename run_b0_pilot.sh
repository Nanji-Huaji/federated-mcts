#!/bin/bash
# Todo 8 — B0 20-ID validation pilot (real endpoints, no mock).
#
# Establishes the exact-arithmetic Game24 baseline on the canonical pilot IDs so
# the B0 measured-evidence projection can be computed from real usage.
#
# Usage: bash run_b0_pilot.sh [extra fmcts-run flags]
#
# NOTE: "$@" is forwarded so flags such as --dry_run take effect.  Omitting it
# silently drops them, which previously turned an intended dry run into a real
# API-spending run.
set -euo pipefail
cd /home/tiantianyi/code/federated-mcts

# Load API credentials (.env is gitignored; keys are never echoed).
set -a
# shellcheck disable=SC1091
. ./.env
set +a

export PYTHONPATH=src

exec .venv/bin/python -u -m federated_mcts \
  --solve_method tot \
  --task game24 \
  --pilot_ids 3 9 16 19 24 25 28 36 40 42 45 71 77 81 102 104 105 116 117 119 \
  --model_config configs/model_config_new_endpoint.json \
  --game24_decision_policy local_validity \
  --game24_oracle_audit \
  --method_generate propose \
  --method_evaluate vote \
  --method_select greedy \
  --n_generate_sample 2 \
  --n_evaluate_sample 3 \
  --n_select_sample 2 \
  --temperature 0.9 \
  --seed 17 \
  --check_format \
  --max_cost_usd 25 \
  "$@"
