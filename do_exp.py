import os
import subprocess
import json

log_file = "logs/game24/naive/deepseek-v3/0.7_propose_n_generate_sample_1_value_n_evaluate_sample_1_method_select_greedy_n_select_sample_1_start0_end148_smg_False_sme_False_check_True_rule_False_warm_True_last_False_idx_0.json"


def do_exp():
    subprocess.run(["zsh", "scripts/game24/eval_deepseek_pre_baseline.sh"])

def rename(file_name, idx):
    """重命名文件"""
    log_dir = os.path.dirname(file_name)
    new_file_name = f"{idx}.json"
    new_path = os.path.join(log_dir, new_file_name)
    
    # 检查原文件是否存在
    if os.path.exists(file_name):
        os.rename(file_name, new_path)
        print(f"Renamed {file_name} to {new_path}")
    else:
        print(f"Warning: File {file_name} does not exist")

def merge_logs(log_dir):
    """合并所有编号的日志文件"""
    def merge_data(data1: dict, data2: dict):
        """合并两个数据字典"""
        for key, value in data2.items():
            if key in ["idx"]:
                continue
            elif key == "ys":
                data1[key].extend(value)
            elif key == "infos":
                data1[key].extend(value)
            elif key == "usage_so_far":
                for subkey in data1[key]:
                    data1[key][subkey] += value[subkey]
            elif key == "time_consumption":
                for subkey in data1[key]:
                    data1[key][subkey] += value[subkey]
        return data1
    
    merged_log = {}
    json_files = []
    
    # 收集所有编号的json文件
    for file_name in os.listdir(log_dir):
        if file_name.split('.')[0].isdigit() and file_name.endswith(".json"):
            json_files.append(file_name)
    
    # 按编号排序
    json_files.sort(key=lambda x: int(x.split('.')[0]))
    
    print(f"Found {len(json_files)} files to merge: {json_files}")
    
    # 逐个读取并合并
    for file_name in json_files:
        file_path = os.path.join(log_dir, file_name)
        try:
            with open(file_path, 'r') as f:
                log = json.load(f)
                if not merged_log:
                    merged_log = log
                else:
                    merged_log = merge_data(merged_log, log)
            print(f"Merged {file_name}")
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
    
    # 保存合并结果
    if merged_log:
        merged_path = os.path.join(log_dir, "merged.json")
        with open(merged_path, 'w') as f:
            json.dump(merged_log, f, indent=4)
        print(f"Merged log saved to {merged_path}")
    else:
        print("No data to merge")

def main():
    """主函数：运行10次实验并合并结果"""
    log_dir = os.path.dirname(log_file)
    
    for idx in range(10):
        print(f"Running experiment {idx + 1}/10...")
        do_exp()
        rename(log_file, idx)
    
    print("All experiments completed. Merging logs...")
    merge_logs(log_dir)
    print("Done!")

if __name__ == "__main__":
    main()
