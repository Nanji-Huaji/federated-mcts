#!/usr/bin/env bash
# Wait for the 20-ID pilot to finish, then launch the 200-ID full validation.
#
# Chained (rather than started together) because the pilot's artifacts ARE the B0
# measured evidence the full run validates against, so the pilot must be complete
# and verifiable first.
set -uo pipefail
cd /home/tiantianyi/code/federated-mcts

LOG=logs/game24/b0_pilot/pilot7.log
FULL_LOG=logs/game24/b0_pilot/full_validation.log
PREFIX="logs/game24/tot/qwen3.7-plus/0.9_propose_n_generate_sample_2_vote_n_evaluate_sample_3_method_select_greedy_n_select_sample_2_start900_end1000_smg_False_sme_False_check_True_rule_False_warm_False_last_False_idx_0"
PILOT_PREFIX="${PREFIX}_ntasks20"

echo "[chain] waiting for pilot to finish..."
for _ in $(seq 1 240); do
  if grep -q "^EXIT=" "$LOG" 2>/dev/null; then break; fi
  sleep 15
done

if ! grep -q "^EXIT=0" "$LOG" 2>/dev/null; then
  echo "[chain] ABORT: pilot did not exit 0; not starting full validation."
  grep -E "^EXIT=" "$LOG" 2>/dev/null || echo "[chain] (no EXIT line; timed out waiting)"
  exit 1
fi
echo "[chain] pilot exited 0."

# The full run validates against the pilot's artifacts as B0 evidence, so refuse to
# start unless the evidence set is complete and internally consistent.
for f in manifest.json performance.json journal.json budget.jsonl; do
  if [ ! -f "${PILOT_PREFIX}_${f}" ]; then
    echo "[chain] ABORT: pilot evidence missing ${PILOT_PREFIX}_${f}"
    exit 1
  fi
done
echo "[chain] pilot evidence present (manifest/performance/journal/budget)."

echo "[chain] launching 200-ID full validation..."
: > "$FULL_LOG"
tmux new-session -d -s b0full \
  "bash run_b0_full_validation.sh >> '$FULL_LOG' 2>&1; echo \"EXIT=\$?\" >> '$FULL_LOG'"
echo "[chain] full validation started in tmux session b0full."
