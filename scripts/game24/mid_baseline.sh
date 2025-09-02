python merged_run.py \
    --task game24 \
    --task_start_index 500 \
    --task_end_index 530 \
    --model_config model_config_deepseek.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method tot \
    --n_evaluate_sample 1 \
    --n_select_sample 10 \
    --warm_start \
    --check_format \
    ${@}
