import re
import os
import sympy
import pandas as pd
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.game24 import *  # type: ignore
from tot.models import gpt
from tot.pattern_match import check_final_result
from tot.pattern_match import check_and_fix_last_line


def get_current_numbers(y: str) -> str:
    last_line = y.strip().split("\n")[-1]
    return last_line.split("left: ")[-1].split(")")[0]


class Game24Task(Task):
    """
    Input (x)   : a string of 4 numbers
    Output (y)  : a trajectory of 3 steps to reach 24
    Reward (r)  : 0 or 1, depending on whether the trajectory is correct
    Input Example:
        1 2 3 4
    Output Example:
        1 + 2 = 3 (left: 3 3 4)
        3 + 3 = 6 (left: 4 6)
        6 * 4 = 24 (left: 24)
        (1 + 2 + 3) * 4 = 24
    """

    def __init__(self, file="24.csv"):
        """
        file: a csv file (fixed)
        """
        super().__init__()
        path = os.path.join(DATA_PATH, "24", file)
        self.data = list(pd.read_csv(path)["Puzzles"])
        self.value_cache = {}
        self.steps = 4
        self.stops = ["\n"] * 4

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        return self.data[idx]

    def test_output_modfiy(self, idx: int, output: str):
        problem_numbers = re.findall(r"\d+", self.data[idx])
        x = problem_numbers[0] + " " + problem_numbers[1] + " " + problem_numbers[2] + " " + problem_numbers[3]
        split_output = output.split("\n")
        output_list = list(filter(None, split_output))
        new_output = ""
        for idx_o, line in enumerate(output_list):
            if idx_o == 0:
                correct, cali_output = check_final_result(line, x=x)
            else:
                correct, cali_output = check_final_result(line, output_list[idx_o - 1])
            if correct == False:
                return {"r": 0}, output
            new_output = new_output + cali_output + "\n"
        if "(left: 24)" in output:
            return {"r": 1}, new_output
        else:
            return {"r": 0}, new_output

    def test_output(self, idx: int, output: str):
        expression = output.strip().split("\n")[-1].lower().replace("answer: ", "").split("=")[0]
        print("expression: ", idx, output, expression)
        numbers = re.findall(r"\d+", expression)
        problem_numbers = re.findall(r"\d+", self.data[idx])
        if sorted(numbers) != sorted(problem_numbers):
            return {"r": 0}
        try:
            # print(sympy.simplify(expression))
            return {"r": int(sympy.simplify(expression) == 24)}
        except Exception as e:
            # print(e)
            return {"r": 0}

    @staticmethod
    def test_output_usingLLM(output: str) -> str:
        input = output
        result = gpt(
            evaluate_the_result.replace("{input}", input),
            n=1,
            stop=None,
            model="gpt-4o",
            api_base="https://try-chatapi.com/v1",
            api_key=os.environ.get("OPENAI_API_KEY"),
        )[0]
        return result

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        current_numbers = get_current_numbers(y if y else x)
        if current_numbers == "24":
            prompt = cot_prompt.format(input=x) + "Steps:" + y
            # print([prompt])
        else:
            # prompt = propose_prompt.format(input=current_numbers)
            numbers = current_numbers.split(" ")
            if len(numbers) == 2:
                prompt = propose_prompt_backup_s2.format(input=current_numbers)
            if len(numbers) == 3:
                prompt = propose_prompt_backup_s1.format(input=current_numbers)
            else:
                prompt = propose_prompt_backup_s0.format(input=current_numbers)
        return prompt

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        last_line = y.strip().split("\n")[-1]
        if "left: " not in last_line or "(left: 24)" in last_line or "Answer: " in last_line:  # last step
            ans = last_line.lower().replace("answer: ", "")
            # print([value_last_step_prompt.format(input=x, answer=ans)])
            return value_last_step_prompt.format(input=x, answer=ans)
        current_numbers = get_current_numbers(y)
        return value_prompt.format(input=current_numbers)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        # if len(y.strip().split("\n")) == 4 and "answer" not in y.lower():
        #     return 0
        value_names = [_.split("\n")[-1] for _ in value_outputs]
        value_map = {"impossible": 0.001, "likely": 1, "sure": 20}  # TODO: ad hoc
        value = sum(value * value_names.count(name) for name, value in value_map.items())
        return value

    @staticmethod
    def pre_generate_check(y):  # Whether it needs to generate
        pattern_final = r"\(left: -?\d+\)"
        if re.search(pattern_final, y):  # reach the final step, no need to generate new proposals
            return False
        return True

    @staticmethod
    def process_generate_result(pro, x, y, check_format):
        if pro.strip() == "":
            return False, pro
        pro = re.sub(r"^[^0-9]+|[^0-9)]+$", "", pro)
        pro = pro.strip()
        if pro != "":
            new_proposal = y + pro + "\n"
        else:
            new_proposal = y
        if check_format == False:
            is_correct, updated_new_proposal = True, new_proposal
        else:
            is_correct, updated_new_proposal = check_and_fix_last_line(new_proposal, x)
        return is_correct, updated_new_proposal

    @staticmethod
    def pre_value_check(y, eval_rule):  # Whether it needs to generate the value
        pattern = [
            "(left: 1 24)",
            "(left: 2 12)",
            "(left: 3 8)",
            "(left: 4 6)",
            "(left: 4 20)",
            "(left: 6 30)",
            "(left: 12 12)",
        ]

        pattern_final = r"\(left: -?\d+\)"
        value, final = 0, False
        if eval_rule == True:  # get the value with some rules
            if "(left: 24)" in y:
                value = 20 + 2
            elif re.search(pattern_final, y):
                final = True
            else:
                for pat in pattern:
                    if pat in y:
                        value = 20 + 1
                        break
        return value, final
