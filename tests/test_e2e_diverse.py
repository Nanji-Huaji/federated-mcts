import sys
import threading
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from federated_mcts.federation.orchestrator import FederatedSolver
from federated_mcts.federation.task_assign import RoundRobinStrategy


class _Args:
    method_generate = "propose"
    method_evaluate = "value"
    method_select = "greedy"
    n_generate_sample = 2
    n_evaluate_sample = 1
    n_select_sample = 2
    check_format = True
    prompt_sample = "standard"
    eval_rule = False
    search_policy = "diverse"
    diverse_joint_rank = True
    diversity_weight = 0.5
    beam_expand = 1
    beam_uncertainty_margin = 0.1
    beam_confidence_margin = 0.8


class _Task:
    steps = 2
    stops = [None, None]
    value_cache = {}

    @staticmethod
    def get_input(idx):
        return "problem"

    @staticmethod
    def pre_generate_check(y):
        return not y.startswith("win")

    @staticmethod
    def propose_prompt_wrap(x, y):
        return "[GENERATE]" + y

    @staticmethod
    def process_generate_result(pro, x, y, check):
        return True, pro.strip()

    @staticmethod
    def canonical_state_key(x, y):
        return y.split(":", 1)[0]

    @staticmethod
    def joint_rank_prompt_wrap(x, candidates):
        return "[JOINT_RANK]" + "|".join(candidates)

    @staticmethod
    def is_success_state(x, y):
        return y.startswith("win")

    @staticmethod
    def value_prompt_wrap(x, y):
        return y

    @staticmethod
    def value_outputs_unwrap(x, y, outputs):
        return 0.0

    @staticmethod
    def pre_value_check(y, eval_rule):
        return 0, False


class _Client:
    def __init__(self, generated):
        self.generated = generated
        self.joint_calls = 0
        self.lock = threading.Lock()

    def __call__(self, args, prompt, n=1, stop=None, **kwargs):
        if prompt.startswith("[JOINT_RANK]"):
            with self.lock:
                self.joint_calls += 1
            return ['{"ranking":[{"id":0,"score":0.9},{"id":1,"score":0.8},{"id":2,"score":0.7}]}']
        return [self.generated]


class TestDiverseFederatedE2E(unittest.TestCase):
    def test_deduplicates_before_one_joint_evaluation_and_stops_on_success(self):
        solver = FederatedSolver.__new__(FederatedSolver)
        solver.args = _Args()
        solver.time_tracker = MagicMock()
        solver._strategy_cache = {}
        local = _Client("same:path-local\nother:path")
        remote = _Client("same:path-remote\nwin:path")
        solver.gpts = {"local": local, "remote": remote}

        ys, info = solver.federated_solve(
            _Task(), 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="remote"),
        )

        self.assertEqual(ys, ["win:path"])
        self.assertEqual(len(info["steps"]), 1)
        self.assertEqual(remote.joint_calls, 0)
        self.assertEqual(len(info["steps"][0]["new_ys"]), 3)
        self.assertEqual(info["steps"][0]["search_metrics"]["raw_candidates"], 4)
        self.assertEqual(info["steps"][0]["search_metrics"]["unique_states"], 3)
        same_state_scores = [
            client_info["values"][0]
            for client_info in info["steps"][0]["client_infos"]
        ]
        self.assertEqual(same_state_scores, [0.0, 0.0])

    def test_equivalent_states_share_reward_across_models(self):
        class OneStepTask(_Task):
            steps = 1
            stops = [None]

        solver = FederatedSolver.__new__(FederatedSolver)
        solver.args = _Args()
        solver.time_tracker = MagicMock()
        solver._strategy_cache = {}
        local = _Client("same:path-local\nother:path")
        remote = _Client("same:path-remote\nthird:path")
        solver.gpts = {"local": local, "remote": remote}

        _, info = solver.federated_solve(
            OneStepTask(), 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="remote"),
        )

        scores = [client_info["values"][0] for client_info in info["steps"][0]["client_infos"]]
        self.assertEqual(scores, [0.9, 0.9])

    def test_baseline_mode_does_not_require_diverse_hooks(self):
        args = _Args()
        args.search_policy = "baseline"
        solver = FederatedSolver.__new__(FederatedSolver)
        solver.args = args
        solver.time_tracker = MagicMock()
        solver._strategy_cache = {}
        solver.gpts = {"only": _Client("a\nb")}

        ys, info = solver.federated_solve(
            _Task(), 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="only"),
        )

        self.assertGreaterEqual(len(ys), 1)
        self.assertEqual(len(info["steps"]), 2)


if __name__ == "__main__":
    unittest.main()
