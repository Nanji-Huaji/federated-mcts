import json
import os
from collections import defaultdict


def merge_json_files(directory_path):
    """
    遍历指定目录下的所有{number}.json文件，合并相同idx的数据
    """
    # 用于存储合并后的数据
    merged_data = defaultdict(
        lambda: {
            "idx": None,
            "ys": [],
            "infos": [],
            "usage_so_far": defaultdict(lambda: defaultdict(float)),
            "time_consumption": {"generation": 0.0, "evaluation": 0.0},
        }
    )

    # 遍历目录下的所有文件
    for filename in os.listdir(directory_path):
        if (
            filename.endswith(".json") and filename[:-5].isdigit()
        ):  # 检查是否为{number}.json格式
            file_path = os.path.join(directory_path, filename)
            print(f"Processing file: {filename}")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # 处理每个条目
                for item in data:
                    idx = item["idx"]

                    # 如果是第一次遇到这个idx，初始化
                    if merged_data[idx]["idx"] is None:
                        merged_data[idx]["idx"] = idx

                    # 合并ys列表
                    merged_data[idx]["ys"].extend(item.get("ys", []))

                    # 合并infos列表
                    merged_data[idx]["infos"].extend(item.get("infos", []))

                    # 累加usage_so_far
                    usage = item.get("usage_so_far", {})
                    for model_name, model_usage in usage.items():
                        for key, value in model_usage.items():
                            merged_data[idx]["usage_so_far"][model_name][
                                key
                            ] += value

                    # 累加time_consumption
                    time_consumption = item.get("time_consumption", {})
                    for key, value in time_consumption.items():
                        merged_data[idx]["time_consumption"][key] += value

            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # 转换为普通字典格式
    result = []
    for idx in sorted(merged_data.keys()):
        item = merged_data[idx]
        # 转换usage_so_far回普通字典
        usage_dict = {}
        for model_name, model_usage in item["usage_so_far"].items():
            usage_dict[model_name] = dict(model_usage)

        result.append(
            {
                "idx": item["idx"],
                "ys": item["ys"],
                "infos": item["infos"],
                "usage_so_far": usage_dict,
                "time_consumption": item["time_consumption"],
            }
        )

    return result


def main():
    directory_path = "logs/game24/naive/deepseek-v3"

    if not os.path.exists(directory_path):
        print(f"Directory {directory_path} does not exist!")
        return

    # 合并所有JSON文件
    merged_result = merge_json_files(directory_path)

    # 输出合并后的结果到新文件
    output_file = os.path.join(directory_path, "merged_results.json")
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(merged_result, f, ensure_ascii=False, indent=2)

    print(f"Merged results saved to: {output_file}")
    print(f"Total unique idx values: {len(merged_result)}")

    # 打印一些统计信息
    total_completion_tokens = 0
    total_prompt_tokens = 0
    total_cost = 0.0

    for item in merged_result:
        usage = item.get("usage_so_far", {}).get("deepseek-v3", {})
        total_completion_tokens += usage.get("completion_tokens", 0)
        total_prompt_tokens += usage.get("prompt_tokens", 0)
        total_cost += usage.get("cost", 0.0)

    print(f"\nOverall Statistics:")
    print(f"Total completion tokens: {total_completion_tokens}")
    print(f"Total prompt tokens: {total_prompt_tokens}")
    print(f"Total cost: ${total_cost:.6f}")


if __name__ == "__main__":
    main()
