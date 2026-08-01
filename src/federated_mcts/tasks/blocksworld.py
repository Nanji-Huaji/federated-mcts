"""PlanBench Blocksworld task (generated_basic).

Contract:
- ``x`` is a canonical, parseable problem encoding produced by ``format_x``:
  instance identity, initial support/holding state, goals and per-instance
  depth bound.
- ``y`` is an append-only newline-delimited sequence of canonical actions, one
  action per tree depth.
- ``process_generate_result`` parses exactly one action line, rejects
  commentary / multiple actions / unknown blocks / illegal preconditions,
  applies it deterministically and appends the normalized action only when
  legal.
- ``canonical_state_key`` is the instance id plus the fully replayed symbolic
  environment state. ``is_success_state`` compares the replayed state to the
  goal and never trusts model declarations.
"""

import json
import os

from federated_mcts.prompts.blocksworld import (
    cot_prompt,
    joint_rank_prompt,
    propose_prompt,
    standard_prompt,
    value_prompt,
)
from federated_mcts.tasks.base import Task, DATA_PATH
from federated_mcts.tasks.blocksworld_engine import (
    action_count,
    canonical_key,
    format_action,
    format_x,
    is_goal_satisfied,
    parse_action_line,
    parse_x,
    render_state,
    replay_state,
    step_state,
)

_DATA_FILENAME = "blocksworld.jsonl"


def _blocksworld_data_path(filename: str) -> str:
    repo_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    )
    repo_data = os.path.join(repo_root, "data", "blocksworld", filename)
    if os.path.exists(repo_data):
        return repo_data
    return os.path.join(DATA_PATH, "blocksworld", filename)


class BlocksworldTask(Task):
    """
    Input (x): canonical problem encoding (instance, initial state, goal, bound)
    Output (y): newline-delimited sequence of canonical actions
    Reward (r): 1 if the replayed state satisfies the goal, else 0
    """

    def __init__(self, file=_DATA_FILENAME):
        super().__init__()
        path = _blocksworld_data_path(file)
        self.data = []
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    self.data.append(json.loads(line))
        self.value_cache = {}
        self.steps = max(record["max_steps"] for record in self.data)
        self.stops = ["\n"] * self.steps
        self._current_idx = 0

    def __len__(self) -> int:
        return len(self.data)

    def get_input(self, idx: int) -> str:
        self._current_idx = idx
        return format_x(self.data[idx])

    @staticmethod
    def standard_prompt_wrap(x: str, y: str = "") -> str:
        return standard_prompt.format(input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y: str = "") -> str:
        return cot_prompt.format(input=x) + y

    @staticmethod
    def propose_prompt_wrap(x: str, y: str = "") -> str:
        current_state = render_state(replay_state(parse_x(x), y))
        return propose_prompt.format(input=x, current_state=current_state)

    @staticmethod
    def value_prompt_wrap(x: str, y: str) -> str:
        parsed = parse_x(x)
        current_state = render_state(replay_state(parsed, y))
        goal = "\n".join("- %s" % " ".join(pred) for pred in sorted(parsed["goal"]))
        return value_prompt.format(input=x, current_state=current_state, goal=goal)

    @staticmethod
    def value_outputs_unwrap(x: str, y: str, value_outputs: list) -> float:
        value_names = [_.split("\n")[-1].strip().lower() for _ in value_outputs]
        value_map = {"impossible": 0.001, "unlikely": 0.1, "likely": 1, "sure": 20}
        return sum(value * value_names.count(name) for name, value in value_map.items())

    @staticmethod
    def joint_rank_prompt_wrap(x: str, candidates: list[str]) -> str:
        parsed = parse_x(x)
        rows = []
        for index, candidate in enumerate(candidates):
            state = render_state(replay_state(parsed, candidate)).replace("\n", "; ")
            rows.append("%d: %s" % (index, state))
        return joint_rank_prompt.format(input=x, candidates="\n".join(rows))

    def canonical_state_key(self, x: str, y: str):
        return canonical_key(parse_x(x), replay_state(parse_x(x), y))

    def is_success_state(self, x: str, y: str) -> bool:
        parsed = parse_x(x)
        return is_goal_satisfied(parsed, replay_state(parsed, y))

    def test_output(self, idx: int, output: str):
        """Runner reward: 1 iff deterministic replay satisfies the goal."""
        x = format_x(self.data[idx])
        return {"r": 1 if self.is_success_state(x, output) else 0}

    def test_output_modify(self, idx: int, output: str):
        """Runner contract: (reward dict, output) as used by merged_run.py."""
        return self.test_output(idx, output), output

    def pre_generate_check(self, y) -> bool:
        """Stop generation on success or when the instance depth bound is hit."""
        record = self.data[self._current_idx]
        if action_count(y) >= record["max_steps"]:
            return False
        return not self.is_success_state(format_x(record), y)

    @staticmethod
    def process_generate_result(pro, x, y, check_format):
        """Parse exactly one action line and append it only when legal."""
        if not isinstance(pro, str):
            return False, y
        line = pro.strip()
        if not line or "\n" in line or "\r" in line:
            return False, y
        parsed = parse_x(x)
        action = parse_action_line(line, parsed["blocks"])
        if action is None:
            return False, y
        next_state = step_state(replay_state(parsed, y), action)
        if next_state is None:
            return False, y
        return True, y + format_action(action) + "\n"

    @staticmethod
    def on_no_valid_candidates(y) -> list[str]:
        """An invalid branch produces no candidates and becomes unexpandable."""
        return []

    def pre_value_check(self, y, eval_rule):
        value, final = 0, False
        if eval_rule:
            record = self.data[self._current_idx]
            if self.is_success_state(format_x(record), y):
                value = 20
            elif action_count(y) >= record["max_steps"]:
                final = True
        return value, final
