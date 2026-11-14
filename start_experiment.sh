#!/bin/bash
cd /home/tiantianyi/code/federated-mcts
export PYTHONPATH=src
mkdir -p logs/blocksworld/dqn_oracle_pilot/run8
exec .venv/bin/python -u -m federated_mcts   --task blocksworld --task_split train   --search_policy dqn --dqn_epsilon 1.0   --dqn_oracle_distance_reward --dqn_oracle_distance_scale 0.25   --dqn_jsonl logs/blocksworld/dqn_oracle_pilot/run8/transitions.jsonl   --max_cost_usd 5   --model_config configs/model_config_vllm.json   --method_generate propose --method_evaluate value --method_select greedy   --n_generate_sample 2 --n_evaluate_sample 1 --n_select_sample 2   --check_format   > logs/blocksworld/dqn_oracle_pilot/run8/collection.log 2>&1
