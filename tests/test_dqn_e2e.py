"""Offline end-to-end contract for search_policy=dqn.

A federated solve driven entirely by fake model clients (no network) must,
under search_policy=dqn, collect one transition per decision step, mark
exactly the final transition done with the terminal correctness reward, and
leave baseline/diverse modes carrying no DQN artifacts.
"""

import sys
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from federated_mcts.federation.orchestrator import FederatedSolver
from federated_mcts.federation.task_assign import RoundRobinStrategy


class _Task:
    steps = 2
    stops = [None, None]
    value_cache = {}

    @staticmethod
    def get_input(idx):
        return "1 2 3 4"

    @staticmethod
    def pre_generate_check(y):
        return not y.endswith("(left: 24)\n")

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
        return 0.5

    @staticmethod
    def pre_value_check(y, eval_rule):
        return 0, False


class _Client:
    """Fake model client: serves generation and evaluation locally, records
    every prompt, and only reveals a success path after the first step."""

    def __init__(self):
        self.calls = []
        self.rank_calls = 0

    def __call__(self, args, prompt, n=1, stop=None, **kwargs):
        self.calls.append(prompt)
        if prompt.startswith("[JOINT_RANK]"):
            self.rank_calls += 1
            return ['{"ranking":[{"id":0,"score":0.9},{"id":1,"score":0.8},{"id":2,"score":0.7}]}']
        trajectory = prompt[len("[GENERATE]"):]
        if trajectory.startswith("keep:"):
            return ["win:path\nnext-extra"]
        return ["keep:s1\nother:s2"]


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
    search_policy = "dqn"
    seed = 42
    diverse_joint_rank = True
    diversity_weight = 0.5
    beam_expand = 1
    beam_uncertainty_margin = 0.1
    beam_confidence_margin = 0.8


def _solver(policy="dqn", client_names=("local", "remote")):
    solver = FederatedSolver.__new__(FederatedSolver)
    solver.args = _Args()
    solver.args.search_policy = policy
    solver.time_tracker = MagicMock()
    solver._strategy_cache = {}
    solver.gpts = {name: _Client() for name in client_names}
    return solver


def _solve(solver, eval_client=None):
    return solver.federated_solve(
        _Task(), 0, to_print=False,
        assign_strategy=RoundRobinStrategy(eval_client=eval_client or next(iter(solver.gpts))),
    )


def _all_transitions(info):
    return [entry for step in info["steps"] for entry in step["dqn_transitions"]]


_TRANSITION_KEYS = {"state", "action", "reward", "next_state", "done", "beam", "joint_rank"}


class TestDqnFederatedE2E(unittest.TestCase):
    def test_dqn_mode_collects_transitions_offline_without_network(self):
        """Given fake clients under search_policy=dqn, when federated_solve
        runs, then transitions are collected with the documented fields and no
        network is touched."""
        solver = _solver()

        _, info = _solve(solver, eval_client="remote")

        transitions = _all_transitions(info)
        self.assertGreaterEqual(len(transitions), 1)
        self.assertGreater(len(solver.gpts["local"].calls), 0)
        for transition in transitions:
            self.assertEqual(set(transition), _TRANSITION_KEYS)
            self.assertIn(transition["beam"], (2, 3, 4, 5))
            self.assertIn(transition["joint_rank"], (False, True))
            self.assertTrue(0 <= transition["action"] < 8)
            self.assertIsInstance(transition["state"], list)
            self.assertIsInstance(transition["reward"], (int, float))

    def test_terminal_reward_assigned_only_to_final_transition(self):
        """Given a successful two-step episode, when the episode finishes,
        then exactly one transition is terminal, it is the last one, and the
        non-terminal transitions carry no correctness credit."""
        solver = _solver()

        _, info = _solve(solver, eval_client="remote")

        transitions = _all_transitions(info)
        done_flags = [transition["done"] for transition in transitions]
        self.assertEqual(done_flags.count(True), 1)
        self.assertTrue(done_flags[-1])
        prior_rewards = [transition["reward"] for transition in transitions[:-1]]
        final_reward = transitions[-1]["reward"]
        for prior in prior_rewards:
            self.assertLessEqual(prior, 0.0)
        self.assertGreaterEqual(final_reward, 0.0)
        self.assertGreater(final_reward, max(prior_rewards, default=-1.0))

    def test_baseline_and_diverse_modes_remain_unaffected(self):
        """Given the same fake setup under baseline and diverse policies, when
        federated_solve runs, then both complete and neither carries DQN
        transition artifacts."""
        for policy in ("baseline", "diverse"):
            with self.subTest(policy=policy):
                solver = _solver(policy=policy, client_names=("only",))

                ys, info = _solve(solver, eval_client="only")

                self.assertGreaterEqual(len(ys), 1)
                for step in info["steps"]:
                    self.assertNotIn("dqn_transitions", step)


if __name__ == "__main__":
    unittest.main()
