python merged_run.py \
    --task game24 \
    --task_start_index 900 \
    --task_end_index 901 \
    --model_config model_config.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --solve_method federated_solve \
    --n_evaluate_sample 3 \
    --n_select_sample 5 \
    ${@}
