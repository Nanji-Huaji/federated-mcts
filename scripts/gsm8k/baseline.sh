python run.py \
    --task gsm8k \
    --task_start_index 0 \
    --task_end_index 50 \
    --naive_run \
    --prompt_sample cot \
    --n_generate_sample 10 \
    --model_config model_config_deepseek.json \
    ${@}