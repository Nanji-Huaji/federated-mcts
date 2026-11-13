#!/usr/bin/env bash
# B0 full validation: all 200 ordered validation IDs, seed 17.
#
# Same decision configuration as the completed 20-ID pilot, so the resulting
# decision_policy_fingerprint is identical and the run is a scale-up of the same
# treatment rather than a new condition.
#
# Cost gate: the plan's original embedded "$0.75" ceiling was superseded by the
# measured 200-ID projection of 1.664672 (operator decision recorded in the plan
# and in .omo/start-work/ledger.jsonl before this run started).
set -euo pipefail

# The completed 20-ID pilot artifacts act as the B0 measured evidence for this
# scale-up.  Passing all four --b0_* paths serves two purposes:
#   1. it activates the measured-evidence gate, which defers the static preflight
#      projection (a worst-case bound: 32.225280 for 200 IDs, above the 20.0 gate)
#      to the real measured projection (1.664672);
#   2. it enforces the fail-closed target-identity gate, so this run is proven to
#      use the same decision configuration as the pilot it scales up.
# NOTE: the `_ntasks20` suffix is mandatory.  The pilot writes its artifacts under
# `<prefix>_ntasks20_*` (the prefix encodes the resolved task count so a 20-ID pilot
# and this 200-ID run can never collide); omitting it makes this run look for files
# that do not exist and abort at the B0 gate.
# NOTE: --max_cost_usd must equal the pilot's value; the identity gate compares it
# as a decision field and rejects a mismatch before any client is constructed.
B0_PREFIX="logs/game24/tot/qwen3.7-plus/0.9_propose_n_generate_sample_2_vote_n_evaluate_sample_3_method_select_greedy_n_select_sample_2_start900_end1000_smg_False_sme_False_check_True_rule_False_warm_False_last_False_idx_0_ntasks20"

if [ -f .env ]; then
  set -a
  # shellcheck disable=SC1091
  . ./.env
  set +a
fi

PYTHONPATH=src exec .venv/bin/python -m federated_mcts \
  --solve_method tot \
  --task game24 \
  --task_split val \
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
  --b0_manifest "${B0_PREFIX}_manifest.json" \
  --b0_performance "${B0_PREFIX}_performance.json" \
  --b0_journal "${B0_PREFIX}_journal.json" \
  --b0_budget "${B0_PREFIX}_budget.jsonl" \
  "$@"
