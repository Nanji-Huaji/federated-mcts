import json
import glob
import os


def check_r_value(infos):
    for info in infos:
        if info == {"r": 1}:
            return True
    return False


def calculate_success_rate(file_slm_performance, file_llm_performance):
    with open(
        file_slm_performance, "r", encoding="utf-8"
    ) as f:  # 通过*performance.json文件获取数据
        data_slm = json.load(f)
    with open(file_llm_performance, "r", encoding="utf-8") as f:
        data_llm = json.load(f)
    llm_avg_sum = data_llm["avg_sum"]
    slm_avg_sum = data_slm["avg_sum"]
    success_rate = (
        llm_avg_sum + slm_avg_sum
    ) / 2  # 大小模型平均的成功率就是两者的平均值
    return success_rate


def calculate_success_time(file_slm, file_llm):
    with open(file_slm, "r", encoding="utf-8") as f:
        data_slm = json.load(f)
    with open(file_llm, "r", encoding="utf-8") as f:
        data_llm = json.load(f)
    success_time = 0
    success_sum = 0
    both_ture, slm_ture, llm_ture, both_false = 0, 0, 0, 0
    for slm, llm in zip(data_slm, data_llm):
        slm_infos = slm.get("infos", None)
        llm_infos = llm.get("infos", None)
        slm_res, llm_res = check_r_value(slm_infos), check_r_value(llm_infos)
        if slm_res or llm_res:
            success_time += 1 # 成功次数在两个模型至少有一个成功的时候加一
        if slm_res and llm_res:
            both_ture += 1
        elif slm_res:
            slm_ture += 1
        elif llm_res:
            llm_ture += 1
        else:
            both_false += 1

        slm_ys = slm.get("ys", None)
        llm_ys = llm.get("ys", None)
        accurate_res = []
        for idx in range(max(len(slm_ys), len(llm_ys))):
            if idx < len(slm_ys) and idx < len(slm_infos) and slm_infos[idx]["r"]==1 and slm_ys[idx] not in accurate_res:
                success_sum += 1
            if idx < len(llm_ys) and idx < len(llm_infos) and llm_infos[idx]["r"]==1 and llm_ys[idx] not in accurate_res:
                success_sum += 1

    return success_time, success_sum/len(data_slm), both_ture/len(data_slm), slm_ture/len(data_slm), llm_ture/len(data_slm), both_false/len(data_slm)



def get_file_path(slm, idx, start_index, end_index):
    if slm:
        file_path = f"C:/Users/hua_j/researchToT/tot/tree-of-thought-llm/logs/game24/bartowski/Phi-3-medium-128k-instruct-GGUF_0.9_propose1_value3_greedy5_start{start_index}_end{end_index}_smg_True_sme_True_check_True_rule_True_warm_True_idx_{idx}.json"
    else:
        file_path = f"C:/Users/hua_j/researchToT/tot/tree-of-thought-llm/logs/game24/bartowski/Phi-3-medium-128k-instruct-GGUF_0.9_propose1_value3_greedy5_start{start_index}_end{end_index}_smg_False_sme_False_check_True_rule_True_warm_False_idx_{idx}.json"
    return file_path


def get_file_path_performance(slm, idx, start_index, end_index):
    if slm:
        file_path = f"C:/Users/hua_j/researchToT/tot/tree-of-thought-llm/logs/game24/bartowski/Phi-3-medium-128k-instruct-GGUF_0.9_propose1_value3_greedy5_start{start_index}_end{end_index}_smg_True_sme_True_check_True_rule_True_warm_True_idx_{idx}_performance.json"
    else:
        file_path = f"C:/Users/hua_j/researchToT/tot/tree-of-thought-llm/logs/game24/bartowski/Phi-3-medium-128k-instruct-GGUF_0.9_propose1_value3_greedy5_start{start_index}_end{end_index}_smg_False_sme_False_check_True_rule_True_warm_False_idx_{idx}_performance.json"
    return file_path


def main(idx, start_index, end_index):
    slm = True
    slm_file_path = get_file_path(slm, idx, start_index, end_index)
    llm_file_path = get_file_path(not slm, idx, start_index, end_index)
    success_rate = calculate_success_rate(
        get_file_path_performance(slm, idx, start_index, end_index),
        get_file_path_performance(not slm, idx, start_index, end_index),
    )
    success_time, success_rate, both_ture, slm_ture, llm_ture, both_false = calculate_success_time(slm_file_path, llm_file_path)
    print(f"Success rate: {success_rate}")
    print(f"Success time: {success_time}")
    print(f"Average accuracy: {success_time / (end_index - start_index)}")
    print(f"Both true: {both_ture}")
    print(f"SLM true only: {slm_ture}")
    print(f"LLM true only: {llm_ture}")
    print(f"Both false: {both_false}")

    result = {
        "idx": idx,
        "success_rate": success_rate,
        "success_time": success_time,
        "average_solved_time": success_time / (end_index - start_index),
    }

    # output_file = f"C:/Users/hua_j/researchToT/tot/tree-of-thought-llm/logs/game24/bartowski/result_{idx}_{start_index}_{end_index}.json"
    # with open(output_file, "w", encoding="utf-8") as f:
    #     json.dump(result, f, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    idx = 1
    start_index = 900
    end_index = 909
    main(idx, start_index, end_index)
