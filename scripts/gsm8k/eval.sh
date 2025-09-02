python merged_run.py \
    --task gsm8k \
    --task_start_index 0 \
    --task_end_index 1 \
    --model_config model_config_deepseek.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method tot \
    --n_evaluate_sample 2 \
    --n_select_sample 3 \
    ${@}