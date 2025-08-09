python merged_run.py \
    --task game24 \
    --task_start_index 100 \
    --task_end_index 105 \
    --model_config qwen_model.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method tot \
    --n_evaluate_sample 3 \
    --n_select_sample 7 \
    ${@}
