import os
import re
import json

from federated_mcts.tasks.base import Task, DATA_PATH
from federated_mcts.prompts.folio import *  # type: ignore

_LABEL_ALIASES = {
    "true": "True",
    "false": "False",
    "unknown": "Unknown",
    "uncertain": "Unknown",
}


def extract_label(output: str) -> str | None:
    """Parse the classification from a '#### True' final answer."""
    if "####" not in output:
        return None
    raw = output.split("####")[-1].strip().rstrip(".")
    key = re.sub(r"[^a-zA-Z]", "", raw).lower()
    return _LABEL_ALIASES.get(key)


def get_current_progress(y: str) -> str:
    if not y.strip():
        return ""
    return y.strip()


def _folio_data_path(filename: str) -> str:
    repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    repo_data = os.path.join(repo_root, "data", "folio", filename)
    if os.path.exists(repo_data):
        return repo_data
    return os.path.join(DATA_PATH, "folio", filename)


class FOLIOBaseTask(Task):
    """
    Input (x): FOLIO premises + conclusion
    Output (y): step-by-step deduction ending in #### True/False/Unknown
    Reward (r): 0 or 1 depending on whether the extracted label matches
    """
    source_filter: str | None = None

    def __init__(self, file="folio-train.jsonl"):
        super().__init__()
        path = _folio_data_path(file)
        self.data = []
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                item = json.loads(line.strip())
                if self.source_filter is None or item["source"] == self.source_filter:
                    self.data.append(item)
        self.value_cache = {}
        self.steps = 5
        self.stops = ["\n"] * 5
        self._current_idx = 0

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        self._current_idx = idx
        item = self.data[idx]
        premises = "\n".join(f"- {premise.strip()}" for premise in item["premises"])
        return f"Premises:\n{premises}\n\nConclusion: {item['conclusion'].strip()}"

    def test_output_modify(self, idx: int, output: str):
        r = {"r": 1 if extract_label(output) == self.data[idx]["label"] else 0}
        return r, output

    def test_output(self, idx: int, output: str):
        return {"r": 1 if extract_label(output) == self.data[idx]["label"] else 0}

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        if "####" in y:
            return cot_prompt.format(input=x) + "Steps:\n" + y
        return propose_prompt.format(input=x, current_progress=get_current_progress(y))

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        if "####" in y:
            return value_last_step_prompt.format(input=x, answer=y.strip())
        return value_prompt.format(input=x, current_progress=get_current_progress(y))

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_names = [_.split("\n")[-1].strip().lower() for _ in value_outputs]
        value_map = {"impossible": 0.001, "unlikely": 0.1, "likely": 1, "sure": 20}
        return sum(value * value_names.count(name) for name, value in value_map.items())

    def canonical_state_key(self, x: str, y: str):
        return tuple(line.strip() for line in y.split("\n") if line.strip())

    def is_success_state(self, x: str, y: str) -> bool:
        if "####" not in y:
            return False
        return extract_label(y) == self.data[self._current_idx]["label"]

    @staticmethod
    def joint_rank_prompt_wrap(x: str, candidates: list[str]) -> str:
        rows = [f"{index}: {candidate.strip().splitlines()[-1]}" for index, candidate in enumerate(candidates)]
        return joint_rank_prompt.format(input=x, candidates="\n".join(rows))
    @staticmethod
    def pre_generate_check(y):
        return "####" not in y

    @staticmethod
    def process_generate_result(pro, x, y, check_format):
        stripped = pro.strip()
        if not stripped:
            return False, y
        return True, y + stripped + "\n"


class WikiLogicTask(FOLIOBaseTask):
    source_filter = "WikiLogic"


class HybLogicTask(FOLIOBaseTask):
    source_filter = "HybLogic"
