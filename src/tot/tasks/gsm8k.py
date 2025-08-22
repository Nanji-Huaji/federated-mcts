import re
import os
import json
import sympy
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.gsm8k import *


def get_current_state(y: str) -> dict:
    """Extract current calculation state from trajectory"""
    lines = y.strip().split("\n")
    if not lines:
        return {"step": 0, "calculations": []}

    calculations = []
    for line in lines:
        if "=" in line or "####" in line:
            calculations.append(line)

    return {
        "step": len(calculations),
        "calculations": calculations,
        "has_answer": any("####" in line for line in lines),
    }


def extract_answer(text: str) -> str | None:
    """Extract final numerical answer from text"""
    match = re.search(r"####\s*(\d+(?:\.\d+)?)", text)
    if match:
        return match.group(1)
    return None


class GSM8KTask(Task):
    """
    Input (x)   : a math word problem
    Output (y)  : a trajectory of reasoning steps leading to the answer
    Reward (r)  : 0 or 1, depending on whether the final answer is correct

    Input Example:
        Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning
        and bakes muffins for her friends every day with four. She sells the remainder
        at the farmers' market daily for $2 per fresh duck egg. How much in dollars
        does she make every day at the farmers' market?

    Output Example:
        Step 1: Janet's ducks lay 16 eggs per day.
        Step 2: She uses 3 eggs for breakfast and 4 eggs for muffins, so 3 + 4 = 7 eggs used.
        Step 3: She has 16 - 7 = 9 eggs left to sell.
        Step 4: She sells each egg for $2, so 9 * 2 = $18.
        #### 18
    """

    def __init__(self, file="train.jsonl"):
        """
        file: a jsonl file with GSM8K data
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "gsm8k", file)
        self.data = []
        with open(path, "r") as f:
            for line in f:
                item = json.loads(line)
                self.data.append(item)
        self.value_cache = {}
        self.steps = 5  # Max number of reasoning steps
        self.stops = ["\n####", "\n\n", "Answer:", "####"]

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]["question"]

    def test_output(self, idx: int, output: str):
        """Test if the output contains the correct answer"""
        predicted_answer = extract_answer(output)
        if predicted_answer is None:
            return {"r": 0}

        ground_truth = extract_answer(self.data[idx]["answer"])
        if ground_truth is None:
            return {"r": 0}

        try:
            # Compare numerical values
            pred_value = float(predicted_answer)
            true_value = float(ground_truth)
            return {"r": int(abs(pred_value - true_value) < 1e-5)}
        except:
            # Fallback to string comparison
            return {"r": int(predicted_answer.strip() == ground_truth.strip())}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(question=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(question=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        """Generate prompt for proposing next step"""
        if not y:
            # First step - no calculations yet
            return propose_prompt.format(question=x, state="(nothing calculated yet)")

        # Check if already has final answer
        if "####" in y:
            return cot_prompt.format(question=x) + y

        # For intermediate steps
        return propose_next_step_prompt.format(question=x, trajectory=y)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        """Generate prompt for evaluating current state"""
        if not y:
            # 返回一个prompt，让LLM来判断空轨迹的价值
            return value_prompt.format(question=x, trajectory="(no steps taken yet)")

        # Check if this is a final answer
        if "####" in y:
            answer = y  # Include full trajectory with answer
            return value_last_step_prompt.format(question=x, answer=answer)

        # For intermediate states
        return value_prompt.format(question=x, trajectory=y)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        """Convert value outputs to numerical score"""
        if not value_outputs:
            return 0.001

        # Count different evaluations
        value_counts = {"sure": 0, "likely": 0, "unlikely": 0, "impossible": 0}

        for output in value_outputs:
            output_lower = output.lower()
            for key in value_counts:
                if key in output_lower:
                    value_counts[key] += 1
                    break

        # Map to numerical values (matching Game24 style)
        value_map = {"impossible": 0.001, "unlikely": 0.1, "likely": 1, "sure": 20}

        # Calculate weighted score
        total = sum(value_map[key] * count for key, count in value_counts.items())

        # Normalize by number of outputs
        if value_outputs:
            total = total / len(value_outputs)

        return total
