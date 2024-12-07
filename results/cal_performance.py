import json
import sys 
sys.path.append(".") 
from run_usingLLM import parse_args

if __name__ == "__main__":
    args = parse_args()

    inference_idx = 0
    if args.naive_run:
        file_name = f"./logs/{args.task}/{args.backend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_usingLLM"
    else:
        file_name = f"./logs/{args.task}/{args.backend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_smg_{args.slm_generate}_sme_{args.slm_eval}_check_{args.check_format}_rule_{args.eval_rule}_warm_{args.warm_start}_idx_{args.inference_idx}"
    cnt_avg, cnt_any = 0, 0
    with open(file_name + "_performance.json", 'r', encoding='utf-8') as file:
        res_json = json.load(file) 

    with open(file_name + ".json", 'r', encoding='utf-8') as file:
        data_list = json.load(file) 
        for json_data in data_list:
            ys = json_data['ys']
            infos = json_data["infos"]
            # log main metric
            accs = [info["r"] for info in infos]
            cnt_avg += sum(accs)  # / len(accs) #jinyu: counting the sum
            cnt_any += any(accs)
        n = len(data_list)
        res_json["avg_sum"] = cnt_avg / n
        res_json["acc"] = cnt_any / n
        
    with open(file_name + "_performance.json", "w") as f:
        json.dump(res_json, f, indent=4)