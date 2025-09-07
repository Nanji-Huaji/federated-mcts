import re
import os
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.multilogieval import *


def get_current_step(y: str) -> int:
    """Extract current reasoning step number from the trajectory"""
    if not y.strip():
        return 0
    steps = [line for line in y.strip().split("\n") if line.strip().startswith("Step")]
    return len(steps)


def has_final_answer(y: str) -> bool:
    """Check if the trajectory contains a final answer"""
    return "answer:" in y.lower()


class MultiLogicEvalTask(Task):
    """
    Input (x): logical context + question
    Output (y): step-by-step logical reasoning leading to yes/no answer
    Reward (r): 0 or 1, depending on whether the final answer is correct

    Input Example:
        Context: If A then B. If B then C. A is true.
        Question: Is C true?
    Output Example:
        Step 1: Given A is true and "If A then B", we can deduce B is true
        Step 2: Given B is true and "If B then C", we can deduce C is true
        Step 3: Therefore, C is true
        Answer: yes
    """

    def __init__(self, file="multilogieval.jsonl"):
        """
        file: a jsonl file containing logical reasoning problems
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "multilogieval", file)
        self.data = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    self.data.append(json.loads(line.strip()))
        self.steps = 4  # max reasoning steps
        self.stops = ["\n"] * 4
        self.value_cache = {}

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        item = self.data[idx]
        return f"Context: {item['context']}\nQuestion: {item['question']}"

    def test_output(self, idx: int, output: str):
        """Test if the output gives correct final answer"""
        # Extract final answer from the output
        lines = output.strip().split("\n")
        answer_line = None

        for line in reversed(lines):
            if "answer:" in line.lower():
                answer_line = line.lower()
                break

        if not answer_line:
            return {"r": 0}

        # Extract predicted answer
        predicted = answer_line.split("answer:")[-1].strip()
        if "yes" in predicted:
            predicted_answer = "yes"
        elif "no" in predicted:
            predicted_answer = "no"
        else:
            return {"r": 0}

        correct_answer = self.data[idx]["answer"]
        return {"r": int(predicted_answer == correct_answer)}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        current_step = get_current_step(y)

        if has_final_answer(y):
            prompt = cot_prompt.format(input=x) + "Reasoning:" + y
        elif current_step >= 3:
            prompt = final_answer_prompt.format(input=x, steps=y)
        else:
            prompt = propose_prompt.format(input=x, current_step=current_step + 1)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if has_final_answer(y):
            return value_final_prompt.format(input=x, reasoning=y)
        else:
            return value_prompt.format(input=x, current_reasoning=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """Convert value outputs to numerical scores"""
        value_names = [_.split("\n")[-1].lower() for _ in value_outputs]
        value_map = {"impossible": 0.001, "unlikely": 0.2, "likely": 1, "sure": 20}
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value if value > 0 else 0.1
