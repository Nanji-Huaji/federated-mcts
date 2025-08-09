python merged_run.py \
    --task game24 \
    --task_start_index 150 \
    --task_end_index 200 \
    --model_config model_config_copy.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method tot \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    --check_format \
    ${@}
