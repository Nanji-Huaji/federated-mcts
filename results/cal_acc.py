import json

import sys 
sys.path.append(".") 
from tot.pattern_match import check_final_result
from run_usingLLM import parse_args

def test_output_modfiy(x: str, output: str):
    split_output = output.split('\n')
    output_list = list(filter(None, split_output))
    new_output = ''
    for idx, line in enumerate(output_list):
        if(idx==0): 
            correct, cali_output = check_final_result(line, x=x)
        else:
            correct, cali_output = check_final_result(line, output_list[idx-1])
        if(correct==False):
            return {"r": 0}, output
        new_output = new_output + '\n' + cali_output
    if "(left: 24)" in output:
        return {"r": 1}, new_output
    else:
        return {"r": 0}, new_output

if __name__ == "__main__":
    args = parse_args()

    inference_idx = 0
    if args.naive_run:
        file_name = f"./logs/{args.task}/{args.localbackend}_{args.temperature}_naive_{args.prompt_sample}_sample_{args.n_generate_sample}_start{args.task_start_index}_end{args.task_end_index}_usingLLM"
    else:
        file_name = f"./logs/{args.task}/{args.localbackend}_{args.temperature}_{args.method_generate}{args.n_generate_sample}_{args.method_evaluate}{args.n_evaluate_sample}_{args.method_select}{args.n_select_sample}_start{args.task_start_index}_end{args.task_end_index}_smg_{args.slm_generate}_sme_{args.slm_eval}_check_{args.check_format}_rule_{args.eval_rule}_warm_{args.warm_start}_idx_{args.inference_idx}"
    with open(file_name + ".json", 'r', encoding='utf-8') as file:
        data_list = json.load(file) 
        for json_data in data_list:
            ys = json_data['ys']
            x = json_data['steps'][0]['x']
            infos = []
            for ys_item in ys:
                res, _ = test_output_modfiy(x, ys_item)
                infos.append(res)
            json_data["infos"] = infos
    
    with open(file_name + ".json", "w") as f:
        json.dump(data_list, f, indent=4)