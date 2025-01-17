import subprocess

python_command = ["python", "run_usingLLM.py"]
python_command_ec = ["python", "run_edge_cloud.py"]
python_command_fed = ["python", "run_federated.py"]

base_command = [
    "--task", "game24",
    "--task_start_index", "900",
    "--task_end_index", "909",
    "--method_generate", "propose",
    "--method_evaluate", "value",
    "--method_select", "greedy",
    "--n_evaluate_sample", "3",
    "--n_select_sample", "5",
    "--temperature", "0.9",
    "--localbackend", "meta-llama-3.1-8b-instruct@q4_k_m",
]
add_command = [
    "--slm_generate",
    "--check_format",
    "--eval_rule",
    "--slm_eval",
    "--warm_start", 
    "--remotebackend", "meta-llama-3.1-8b-instruct@q4_k_m",
    ]


command_list = [
    #python_command + base_command + ["--slm_generate", "--slm_eval"],
    #python_command + base_command + ["--slm_generate", "--slm_eval", "--check_format","--eval_rule"],
    #python_command + base_command + ["--slm_generate", "--slm_eval", "--check_format","--eval_rule","--last_lm"],
    #python_command_ec + base_command + ["--check_format","--eval_rule","--last_lm"],
    #python_command + base_command + ["--slm_generate", "--check_format","--eval_rule","--last_lm"],
    #python_command + base_command + ["--slm_eval", "--check_format","--eval_rule"],
    python_command + base_command + ["--check_format","--eval_rule"],
]



# Loop --inference_idx from 0-2
for command in command_list:
    for inference_idx in range(0,2):
        cur_command = command + ["--inference_idx", str(inference_idx)]
        print(f"Running command: {' '.join(cur_command)}")
        try:
            subprocess.run(cur_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running command with --inference_idx {inference_idx}: {e}")