import re
import os
import pandas as pd
from datasets import load_dataset
from tot.tasks.base import Task, DATA_PATH
from tot.models import gpt

# 导入prompt文件
from tot.prompts.aime import *  # type: ignore


class AIMETask(Task):
    """
    AIME (American Invitational Mathematics Examination) Task

    Input (x)   : A mathematical competition problem statement
    Output (y)  : A step-by-step solution leading to the final answer
    Reward (r)  : 0 or 1, depending on whether the final answer is correct

    Input Example:
        "Find the number of ordered pairs (a,b) of positive integers such that a+b=1000 and neither a nor b has a zero digit."

    Output Example:
        Step 1: We need to count positive integers from 1 to 999 that have no zero digits.
        Step 2: For a d-digit number with no zero digits, each digit can be 1,2,...,9 (9 choices).
        Step 3: 1-digit: 9 numbers, 2-digit: 81 numbers, 3-digit: 729 numbers.
        ...
        Final Answer: 738
    """

    def __init__(self, dataset_name="Maxwell-Jia/AIME_2024"):
        """
        Load AIME 2024 dataset from HuggingFace
        """
        super().__init__()
        # Load the dataset
        ds = load_dataset(dataset_name)
        self.data = []

        # Convert to list format for easier access
        for split in ds:
            for item in ds[split]:
                self.data.append(
                    {
                        "id": item["ID"],
                        "problem": item["Problem"],
                        "solution": item["Solution"],
                        "answer": item["Answer"],
                    }
                )

        self.value_cache = {}
        self.steps = 5
        self.stops = ["\n"] * 10  # AIME solutions typically longer than game24

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]["problem"]

    def get_answer(self, idx: int) -> int:
        """Get the correct answer for a problem"""
        return int(self.data[idx]["answer"])

    def test_output(self, idx: int, output: str):
        """
        Test if the output contains the correct final answer.
        Look for patterns like "Final Answer: XXX" or "Answer: XXX"
        """
        correct_answer = self.get_answer(idx)

        # Extract the final answer from the output
        # Look for common patterns in mathematical solutions
        patterns = [
            r"(?:Final\s+)?Answer:\s*(\d+)",
            r"(?:The\s+)?answer\s+is\s+(\d+)",
            r"Therefore,?\s+(\d+)",
            r"Thus,?\s+(\d+)",
            r"=\s*(\d+)\s*$",  # Ends with = number
        ]

        predicted_answer = None
        for pattern in patterns:
            matches = re.findall(pattern, output, re.IGNORECASE)
            if matches:
                try:
                    predicted_answer = int(matches[-1])  # Take the last match
                    break
                except ValueError:
                    continue

        if predicted_answer is None:
            # Try to extract the last number from the output as a fallback
            numbers = re.findall(r"\b(\d+)\b", output)
            if numbers:
                try:
                    predicted_answer = int(numbers[-1])
                except ValueError:
                    return {"r": 0}
            else:
                return {"r": 0}

        return {"r": int(predicted_answer == correct_answer)}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        safe_x = x.replace("{", "{{").replace("}", "}}")
        return cot_prompt.format(input=safe_x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        if "Final Answer:" in y or "Answer:" in y:
            # Already have the final answer, use cot prompt
            return AIMETask.cot_prompt_wrap(x, y)

        # Determine which backup prompt to use based on progress
        lines = [line.strip() for line in y.split("\n") if line.strip()]
        step_count = len([line for line in lines if "Step" in line])

        if step_count == 0:
            # Just starting - use backup s0
            return propose_prompt_backup_s0.format(input=x)
        elif step_count >= 3:
            # Near completion - use backup s2
            current_progress = x + "\n\nCurrent progress:\n" + y
            return propose_prompt_backup_s2.format(input=current_progress)
        else:
            # Middle stage - use backup s1
            current_progress = x + "\n\nCurrent progress:\n" + y
            return propose_prompt_backup_s1.format(input=current_progress)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if "Final Answer:" in y or "Answer:" in y:
            # This is a complete solution, use last step prompt
            return value_last_step_prompt.format(input=x, answer=y)
        else:
            # This is a partial solution, evaluate progress
            current_state = x + "\n\nCurrent progress:\n" + y
            return value_prompt.format(input=current_state)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_names = [_.split("\n")[-1].strip() for _ in value_outputs]
        value_map = {"impossible": 0.001, "likely": 1, "sure": 20}
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value

    @staticmethod
    def pre_generate_check(y):
        """Check if we need to continue generating"""
        # Stop if we have a final answer
        return not ("Final Answer:" in y or "Answer:" in y)

    @staticmethod
    def process_generate_result(pro, x, y, check_format=False):
        """Process the generated proposal"""
        if pro.strip() == "":
            return False, pro

        # Clean up the proposal
        pro = pro.strip()
        new_proposal = y + pro + "\n"

        # Basic format check - ensure it's a reasonable mathematical step
        if check_format:
            # Very basic check - contains some mathematical content
            if not any(char in pro for char in "0123456789=+-*/()[]{}"):
                return False, new_proposal

        return True, new_proposal

    def test_output_modify(self, idx: int, output: str):
        """No modification needed for AIME"""
        return self.test_output(idx, output)

    @staticmethod
    def pre_value_check(y, eval_rule=True):
        """Check if we need to evaluate this solution"""
        if "Final Answer:" in y or "Answer:" in y:
            return 20, True  # High value for complete solutions

        # Count progress indicators
        steps = len([line for line in y.split("\n") if line.strip() and "Step" in line])
        if steps > 0:
            return 1 + steps * 0.5, False  # Progressive value for partial solutions

        return 0, False
