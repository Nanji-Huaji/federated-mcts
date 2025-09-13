import subprocess
import os

n_select_samples = [i for i in range(1, 20 + 1)]

cmd = """
python merged_run.py \
    --task game24 \
    --task_start_index 0 \
    --task_end_index 49 \
    --model_config model_config_deepseek.json \
    --method_generate propose \
    --method_evaluate value \
    --method_select greedy \
    --temperature 0.7 \
    --solve_method speculative_solve \
    --n_evaluate_sample 1 \
    --n_select_sample 10 \
    --warm_start \
    --check_format \
    --use_hardest_data \
    ${@}
"""


def run_exp(n_select_sample):
    full_cmd = cmd.replace("${@}", f"--n_select_sample {n_select_sample}")
    print(f"Running command: {full_cmd}")
    subprocess.run(full_cmd, shell=True, check=True)


if __name__ == "__main__":
    for n in n_select_samples:
        run_exp(n)
