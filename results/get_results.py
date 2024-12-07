import argparse, json, glob

import sys 
sys.path.append(".") 
from run_usingLLM import parse_args

if __name__ == "__main__":
    args = parse_args()

    avg_sum, acc = [], []
    lat_all, lat_generate, lat_eval = [], [], []
    slm_completion_tokens, slm_prompt_tokens, llm_completion_tokens, llm_prompt_tokens = [], [], [], []
    cost =[]

    data_list = []
    
    if args.naive_run:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_usingLLM"
    else:
        file = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_smg_{args.slm_generate}_sme_{args.slm_eval}_check_{args.check_format}_rule_{args.eval_rule}_warm_{args.warm_start}_idx_*"
    file_pattern = (file +'_performance.json')
    matching_files = glob.glob(file_pattern)

    for file_name in matching_files:
        with open(file_name, 'r', encoding='utf-8') as file:
            data = json.load(file)  # 将 JSON 内容解析为 Python 对象
            data_list.append(data)

    averages = {}
    for key in data_list[0].keys():
        # 求每个键的平均值
        averages[key] = sum(d[key] for d in data_list) / len(data_list)

    print('Final result: ', averages)