import re
import os
import json
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.gsm8k import *


def extract_answer(text: str) -> str:
    """从答案文本中提取最终数值"""
    # 寻找 #### 格式的答案
    if "####" in text:
        return text.split("####")[-1].strip()

    # 寻找最后一个数字
    numbers = re.findall(r"-?\d+\.?\d*", text)
    return numbers[-1] if numbers else ""


def get_current_progress(y: str) -> str:
    """获取当前推理进度"""
    if not y.strip():
        return ""
    lines = y.strip().split("\n")
    return "\n".join(lines)


class GSM8KTask(Task):
    """
    Input (x): 一个数学问题的字符串
    Output (y): 逐步解决问题的推理轨迹
    Reward (r): 0 或 1，根据最终答案是否正确

    Input Example:
        "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast..."
    Output Example:
        Janet eats 3 eggs and uses 4 for muffins
        So she has 16 - 3 - 4 = 9 eggs left to sell
        She makes 9 * 2 = 18 dollars
        #### 18
    """

    def __init__(self, file="gsm8k.jsonl"):
        """
        file: jsonl文件，包含问题和答案
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "gsm8k", file)
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                self.data.append(item)

        self.value_cache = {}
        self.steps = 6  # GSM8K问题通常需要更多步骤
        self.stops = ["\n"] * 6

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]["question"]

    def test_output(self, idx: int, output: str):
        """测试输出是否正确"""
        try:
            # 提取预测答案
            predicted_answer = extract_answer(output)
            # 提取真实答案
            true_answer = extract_answer(self.data[idx]["answer"])

            if not predicted_answer or not true_answer:
                return {"r": 0}

            # 转换为数值比较
            pred_num = float(predicted_answer.replace(",", ""))
            true_num = float(true_answer.replace(",", ""))

            # 允许小的数值误差
            tolerance = max(0.01, abs(true_num) * 0.001)  # 相对误差0.1%或绝对误差0.01
            is_correct = abs(pred_num - true_num) <= tolerance

            return {"r": 1 if is_correct else 0}

        except Exception as e:
            print(f"Error in test_output: {e}")
            return {"r": 0}

    def test_output_modify(self, idx: int, output: str):
        return self.test_output(idx, output)

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        current_progress = get_current_progress(y)
        if "####" in y:  # 已完成
            prompt = cot_prompt.format(input=x) + "Steps:\n" + y
        else:
            prompt = propose_prompt.format(input=x, current_progress=current_progress)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if "####" in y:  # 最后一步，评估完整解答
            return value_last_step_prompt.format(input=x, answer=y.strip())
        else:
            current_progress = get_current_progress(y)
            return value_prompt.format(input=x, current_progress=current_progress)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # 如果没有最终答案，评分较低
        if "####" not in y and len(y.strip().split("\n")) >= 5:
            return 0.1

        value_names = [_.split("\n")[-1].strip().lower() for _ in value_outputs]
        value_map = {"impossible": 0.001, "unlikely": 0.1, "likely": 1, "sure": 20}

        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value
