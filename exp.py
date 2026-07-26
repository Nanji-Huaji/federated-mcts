import json
import subprocess
import os
import time
import warnings
from typing import List, Dict, Any
from datetime import datetime

"""
探究实验脚本：
1. 在n_evaluate_sample=1时，探索不同n_select_sample的性能
2. 选择正确率最高的三个参数配置
3. 在更大任务范围(900-930)上测试这些配置
4. 运行baseline(tot)进行对比
5. 输出结果到json和markdown文件
"""


def run_experiment(config: Dict[str, Any]) -> bool:
    """运行单个实验"""
    command = [
        "python",
        "merged_run.py",
        "--task",
        config["task"],
        "--task_start_index",
        str(config["task_start_index"]),
        "--task_end_index",
        str(config["task_end_index"]),
        "--model_config",
        config["model_config"],
        "--method_generate",
        config["method_generate"],
        "--method_evaluate",
        config["method_evaluate"],
        "--method_select",
        config["method_select"],
        "--temperature",
        str(config["temperature"]),
        "--solve_method",
        config["solve_method"],
        "--n_evaluate_sample",
        str(config["n_evaluate_sample"]),
        "--n_select_sample",
        str(config["n_select_sample"]),
    ]

    if config.get("check_format"):
        command.append("--check_format")

    print(f"Running: {' '.join(command)}")
    result = subprocess.run(command, capture_output=True, text=True)

    if result.returncode != 0:
        print(f"Error: {result.stderr}")
        return False
    return True


def get_file_path(config: Dict[str, Any]) -> str:
    """生成结果文件路径"""
    try:
        with open(config["model_config"], "r") as f:
            model_config_data = json.load(f)

        remotebackend = next(
            (model["model"] for model in model_config_data if model["client_name"] == "remote_client"), "unknown_remote"
        )

        # 构建文件路径
        file_path = (
            f"./logs/{config['task']}/{config['solve_method']}/{remotebackend}/"
            f"{config['temperature']}_{config['method_generate']}_n_generate_sample_"
            f"{config['n_select_sample']}_{config['method_evaluate']}_n_evaluate_sample_"
            f"{config['n_evaluate_sample']}_method_select_{config['method_select']}_"
            f"n_select_sample_{config['n_select_sample']}_start{config['task_start_index']}_"
            f"end{config['task_end_index']}_smg_False_sme_False_check_{config.get('check_format', False)}_"
            f"rule_False_warm_False_last_False_idx_0"
        )

        return file_path
    except Exception as e:
        print(f"Error generating file path: {e}")
        return None


def get_performance(file_path: str, model_config_file: str) -> Dict[str, float]:
    """获取实验性能指标"""
    performance = {"solve_rate": 0.0, "avg_solve_count": 0.0, "cost": 0.0}
    performance_file = file_path + "_performance.json"

    if not os.path.exists(performance_file):
        warnings.warn(f"Performance file not found: {performance_file}", UserWarning)
        print(f"WARNING: Performance file not found: {performance_file}")
        return performance

    try:
        with open(performance_file, "r") as f:
            performance_dict = json.load(f)
            performance["solve_rate"] = performance_dict.get("acc", 0.0)
            performance["avg_solve_count"] = performance_dict.get("avg_sum", 0.0)

            # 获取cost信息
            with open(model_config_file, "r") as model_f:
                model_config = json.load(model_f)
                remote_model = next(
                    (model["model"] for model in model_config if model["client_name"] == "remote_client"), "deepseek-v3"
                )
                performance["cost"] = performance_dict.get(remote_model, {"cost": 0.0}).get("cost", 0.0)

    except Exception as e:
        warnings.warn(f"Error reading performance file: {e}", UserWarning)
        print(f"WARNING: Error reading performance file: {e}")

    return performance


def run_exploration_phase() -> List[Dict[str, Any]]:
    """阶段1: 探索不同n_select_sample的性能 (n_evaluate_sample=1)"""
    print("\n" + "=" * 80)
    print("PHASE 1: Exploring different n_select_sample values")
    print("=" * 80)

    # 探索配置：n_evaluate_sample=1，不同的n_select_sample
    exploration_configs = []
    n_select_samples = [5, 8, 10, 12, 15, 18, 20, 25, 30]

    for n_select in n_select_samples:
        config = {
            "task": "game24",
            "task_start_index": 900,
            "task_end_index": 910,  # 小范围先测试
            "model_config": "configs/model_config_deepseek.json",
            "method_generate": "propose",
            "method_evaluate": "value",
            "method_select": "greedy",
            "temperature": 0.7,
            "solve_method": "speculative_solve",
            "n_evaluate_sample": 1,  # 固定为1
            "n_select_sample": n_select,
            "check_format": True,
        }
        exploration_configs.append(config)

    results = []
    for i, config in enumerate(exploration_configs):
        print(f"\nRunning exploration {i+1}/{len(exploration_configs)}: n_select_sample = {config['n_select_sample']}")

        start_time = time.time()
        success = run_experiment(config)
        end_time = time.time()

        # 只记录配置和运行状态，不读取性能文件
        results.append(
            {
                "config": config,
                "performance": None,  # 稍后批量读取
                "runtime": end_time - start_time,
                "success": success,
            }
        )

        if success:
            print(f"Experiment completed successfully")
        else:
            print(f"Experiment failed")

    return results


def get_top_configurations(exploration_results: List[Dict[str, Any]], top_k: int = 3) -> List[Dict[str, Any]]:
    """获取正确率最高的top_k个配置"""
    successful_results = [r for r in exploration_results if r["success"] and r["performance"] is not None]
    if not successful_results:
        print("No successful experiments found!")
        return []

    # 按solve_rate排序
    sorted_results = sorted(successful_results, key=lambda x: x["performance"]["solve_rate"], reverse=True)
    top_configs = [r["config"] for r in sorted_results[:top_k]]

    print(f"\nTop {top_k} configurations by solve rate:")
    for i, result in enumerate(sorted_results[:top_k]):
        config = result["config"]
        perf = result["performance"]
        print(
            f"{i+1}. n_select_sample={config['n_select_sample']}: "
            f"solve_rate={perf['solve_rate']:.3f}, cost=${perf['cost']:.4f}"
        )

    return top_configs


def run_full_evaluation(top_configs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """阶段2: 在完整任务集上评估top配置 + baseline"""
    print("\n" + "=" * 80)
    print("PHASE 2: Full evaluation on tasks 900-930")
    print("=" * 80)

    full_configs = []

    # 1. 扩展top配置到完整任务范围
    for config in top_configs:
        full_config = config.copy()
        full_config["task_start_index"] = 900
        full_config["task_end_index"] = 930
        full_configs.append(full_config)

    # 2. 添加baseline配置 (ToT)
    baseline_config = {
        "task": "game24",
        "task_start_index": 900,
        "task_end_index": 930,
        "model_config": "configs/model_config_deepseek.json",
        "method_generate": "propose",
        "method_evaluate": "value",
        "method_select": "greedy",
        "temperature": 0.7,
        "solve_method": "tot",  # baseline使用tot
        "n_evaluate_sample": 1,
        "n_select_sample": 10,  # 使用标准值
        "check_format": True,
    }
    full_configs.append(baseline_config)

    results = []
    for i, config in enumerate(full_configs):
        is_baseline = config["solve_method"] == "tot"
        exp_type = "Baseline (ToT)" if is_baseline else f"Top config {i+1}"

        print(
            f"\nRunning {exp_type}: "
            f"solve_method={config['solve_method']}, n_select_sample={config['n_select_sample']}"
        )

        start_time = time.time()
        success = run_experiment(config)
        end_time = time.time()

        # 只记录配置和运行状态，不读取性能文件
        results.append(
            {
                "config": config,
                "performance": None,  # 稍后批量读取
                "runtime": end_time - start_time,
                "success": success,
                "type": "baseline" if is_baseline else "top_config",
            }
        )

        if success:
            print(f"Experiment completed successfully")
        else:
            print(f"Experiment failed")

    return results


def batch_read_performance(results: List[Dict[str, Any]], wait_time: int = 10) -> List[Dict[str, Any]]:
    """批量读取所有实验的性能文件"""
    print("\n" + "=" * 80)
    print(f"READING PERFORMANCE FILES (waiting {wait_time}s for file generation)")
    print("=" * 80)

    # 等待文件生成
    time.sleep(wait_time)

    updated_results = []
    successful_reads = 0
    failed_reads = 0

    for i, result in enumerate(results):
        if result["success"]:
            print(f"\nReading performance file {i+1}/{len(results)}...")
            file_path = get_file_path(result["config"])

            if file_path:
                performance = get_performance(file_path, result["config"]["model_config"])

                # 检查是否成功读取
                if performance["solve_rate"] > 0 or performance["cost"] > 0:
                    successful_reads += 1
                    print(
                        f"✅ Successfully read: solve_rate={performance['solve_rate']:.3f}, cost=${performance['cost']:.4f}"
                    )
                else:
                    failed_reads += 1
                    print(f"⚠️  Warning: Performance file exists but contains zero values")

                result["performance"] = performance
            else:
                failed_reads += 1
                print(f"❌ Failed to generate file path")
                result["performance"] = {"solve_rate": 0.0, "avg_solve_count": 0.0, "cost": 0.0}
        else:
            print(f"\nSkipping failed experiment {i+1}/{len(results)}")
            result["performance"] = {"solve_rate": 0.0, "avg_solve_count": 0.0, "cost": 0.0}

        updated_results.append(result)

    print(f"\n📊 Performance file reading summary:")
    print(f"   Successful reads: {successful_reads}")
    print(f"   Failed reads: {failed_reads}")
    print(f"   Total experiments: {len(results)}")

    if failed_reads > 0:
        warnings.warn(f"Failed to read {failed_reads} performance files out of {len(results)} experiments", UserWarning)

    return updated_results


def save_results(exploration_results: List[Dict[str, Any]], evaluation_results: List[Dict[str, Any]], timestamp: str):
    """保存结果到JSON和Markdown文件"""

    # 保存到JSON
    json_results = {
        "timestamp": timestamp,
        "exploration_phase": exploration_results,
        "evaluation_phase": evaluation_results,
        "summary": {
            "exploration_count": len(exploration_results),
            "evaluation_count": len(evaluation_results),
            "successful_explorations": len([r for r in exploration_results if r["success"]]),
            "successful_evaluations": len([r for r in evaluation_results if r["success"]]),
        },
    }

    json_filename = f"experiment_results_{timestamp}.json"
    with open(json_filename, "w") as f:
        json.dump(json_results, f, indent=2, default=str)

    # 保存到Markdown
    md_filename = f"experiment_report_{timestamp}.md"
    with open(md_filename, "w") as f:
        f.write(f"# Game24 Experiment Report\n\n")
        f.write(f"**Generated on:** {datetime.fromtimestamp(float(timestamp)).strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        f.write("## Experiment Overview\n\n")
        f.write("This experiment explores the performance of different sampling strategies for the Game24 task:\n\n")
        f.write("1. **Phase 1**: Exploration of different `n_select_sample` values with `n_evaluate_sample=1`\n")
        f.write("2. **Phase 2**: Full evaluation of top-3 configurations vs baseline (ToT) on tasks 900-930\n\n")

        # Phase 1 Results
        f.write("## Phase 1: Exploration Results\n\n")
        f.write("| n_select_sample | Solve Rate | Avg Solve Count | Cost ($) | Runtime (s) | Status |\n")
        f.write("|-----------------|------------|-----------------|----------|-------------|--------|\n")

        for result in exploration_results:
            config = result["config"]
            perf = result["performance"]
            status = "✅" if result["success"] else "❌"
            f.write(
                f"| {config['n_select_sample']} | {perf['solve_rate']:.3f} | "
                f"{perf['avg_solve_count']:.3f} | ${perf['cost']:.4f} | "
                f"{result['runtime']:.2f} | {status} |\n"
            )

        # Top configurations
        successful_exploration = [r for r in exploration_results if r["success"] and r["performance"]["solve_rate"] > 0]
        if successful_exploration:
            top_3 = sorted(successful_exploration, key=lambda x: x["performance"]["solve_rate"], reverse=True)[:3]
            f.write(f"\n### Top 3 Configurations\n\n")
            for i, result in enumerate(top_3):
                config = result["config"]
                perf = result["performance"]
                f.write(
                    f"{i+1}. **n_select_sample = {config['n_select_sample']}**: "
                    f"Solve Rate = {perf['solve_rate']:.3f}, Cost = ${perf['cost']:.4f}\n"
                )

        # Phase 2 Results
        f.write(f"\n## Phase 2: Full Evaluation Results (Tasks 900-930)\n\n")
        f.write(
            "| Configuration | Method | n_select_sample | Solve Rate | Avg Solve Count | Cost ($) | Runtime (s) | Status |\n"
        )
        f.write(
            "|---------------|--------|-----------------|------------|-----------------|----------|-------------|--------|\n"
        )

        for i, result in enumerate(evaluation_results):
            config = result["config"]
            perf = result["performance"]
            status = "✅" if result["success"] else "❌"
            config_name = "Baseline (ToT)" if result["type"] == "baseline" else f"Top Config {i+1}"

            f.write(
                f"| {config_name} | {config['solve_method']} | {config['n_select_sample']} | "
                f"{perf['solve_rate']:.3f} | {perf['avg_solve_count']:.3f} | ${perf['cost']:.4f} | "
                f"{result['runtime']:.2f} | {status} |\n"
            )

        # Analysis
        f.write(f"\n## Analysis\n\n")
        successful_eval = [r for r in evaluation_results if r["success"] and r["performance"]["solve_rate"] > 0]
        if successful_eval:
            best_result = max(successful_eval, key=lambda x: x["performance"]["solve_rate"])
            baseline_result = next((r for r in successful_eval if r["type"] == "baseline"), None)

            f.write(f"### Best Performance\n")
            config = best_result["config"]
            perf = best_result["performance"]
            config_type = "Baseline (ToT)" if best_result["type"] == "baseline" else "Speculative Solve"
            f.write(f"- **Method**: {config_type}\n")
            f.write(f"- **n_select_sample**: {config['n_select_sample']}\n")
            f.write(f"- **Solve Rate**: {perf['solve_rate']:.3f}\n")
            f.write(f"- **Cost**: ${perf['cost']:.4f}\n\n")

            if baseline_result and best_result["type"] != "baseline":
                baseline_perf = baseline_result["performance"]
                if baseline_perf["solve_rate"] > 0:
                    improvement = (perf["solve_rate"] - baseline_perf["solve_rate"]) / baseline_perf["solve_rate"] * 100
                    cost_ratio = perf["cost"] / baseline_perf["cost"] if baseline_perf["cost"] > 0 else float("inf")
                    f.write(f"### Comparison with Baseline\n")
                    f.write(f"- **Solve Rate Improvement**: {improvement:+.1f}%\n")
                    f.write(f"- **Cost Ratio**: {cost_ratio:.2f}x\n\n")

        f.write("## Configuration Details\n\n")
        f.write("```json\n")
        f.write("Base Configuration:\n")
        f.write("{\n")
        f.write('  "task": "game24",\n')
        f.write('  "model_config": "configs/model_config_deepseek.json",\n')
        f.write('  "method_generate": "propose",\n')
        f.write('  "method_evaluate": "value",\n')
        f.write('  "method_select": "greedy",\n')
        f.write('  "temperature": 0.7,\n')
        f.write('  "n_evaluate_sample": 1,\n')
        f.write('  "check_format": true\n')
        f.write("}\n")
        f.write("```\n")

    print(f"\nResults saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  Markdown: {md_filename}")


def main():
    """主函数"""
    timestamp = str(int(time.time()))

    print("Game24 Sampling Strategy Experiment")
    print("=" * 50)

    # Phase 1: 探索阶段
    exploration_results = run_exploration_phase()

    # Phase 2: 完整评估 (这里先运行，但还没有performance数据)
    print("\n📋 Preparing for Phase 2...")

    # 为了准备Phase 2，我们需要先读取exploration的结果来获取top配置
    # 这里我们临时读取exploration结果
    print("📊 Reading exploration results to determine top configurations...")
    exploration_results_with_perf = batch_read_performance(exploration_results, wait_time=5)

    # 获取top-3配置
    top_configs = get_top_configurations(exploration_results_with_perf, top_k=3)

    if not top_configs:
        print("❌ No successful configurations found. Exiting...")
        return

    # 运行完整评估
    evaluation_results = run_full_evaluation(top_configs)

    # 批量读取所有性能文件
    print("\n📊 Reading all performance files after all experiments completed...")
    exploration_results_final = batch_read_performance(exploration_results, wait_time=10)
    evaluation_results_final = batch_read_performance(evaluation_results, wait_time=10)

    # 保存结果
    save_results(exploration_results_final, evaluation_results_final, timestamp)

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
