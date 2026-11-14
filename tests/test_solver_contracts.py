"""Contract tests for the solver-facing task interface.

These tests cross the same seams the experiment runner uses:
``finish_task`` / ``FederatedSolver.solve`` / ``FederatedSolver.federated_solve``
with injected fake clients, a run-scoped usage tracker, and a real temporary
filesystem for DQN transition recording.

Covered contracts:
- evaluator value caches are scoped to one solver, one task, one client
  instance and one evaluator identity;
- DQN step resources are the generation wall clock and this run's scoped
  usage delta, not the sum of per-client durations;
- DQN scores map back to the client(s) that proposed them, and candidates the
  session never scored are not reported as 0.0 to the assignment strategy;
- an empty generation branch completes with an explicit frontier selection;
- a returned task outcome is final: its verdict and terminal reward are fixed
  before the caller sees it, and the terminal JSONL record is written once.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
import unittest
from copy import deepcopy

sys.path.insert(0, "src")

from federated_mcts.config import DecisionSpec
from federated_mcts.federation import FederatedSolver
from federated_mcts.federation.federated_execution import attach_dqn_scores
from federated_mcts.federation.outcome import finish_task
from federated_mcts.federation.task_assign import BaseAssignStrategy, RoundRobinStrategy
from federated_mcts.models.usage import get_usage_tracker, usage_scope


# ── Fakes ─────────────────────────────────────────────────────────────


class _ValueTask:
    """Two-step task whose value prompt deliberately omits the puzzle input.

    That is the historical shape that made a prompt-only cache unsafe; it is
    also the shape in which a single evaluator should be charged only once.
    """

    steps = 2
    stops = [None, None]

    @staticmethod
    def get_input(idx):
        return "puzzle"

    @staticmethod
    def standard_prompt_wrap(x, y):
        return "[GENERATE]" + y

    @staticmethod
    def propose_prompt_wrap(x, y):
        return "[PROPOSE]" + y

    @staticmethod
    def process_generate_result(proposal, x, y, check):
        return True, proposal.strip()

    @staticmethod
    def canonical_state_key(x, y):
        return y.strip()

    @staticmethod
    def is_success_state(x, y):
        return "win" in y

    @staticmethod
    def value_prompt_wrap(x, y):
        return "[VALUE]"

    @staticmethod
    def value_outputs_unwrap(x, y, outputs):
        return float(outputs[0]) if outputs else 0.0

    @staticmethod
    def test_output(idx, candidate):
        # The exact verifier disagrees with is_success_state in one test.
        return {"r": 0}


class _OneStepTask(_ValueTask):
    steps = 1
    stops = [None]


class _IndexedInputTask(_ValueTask):
    @staticmethod
    def get_input(idx):
        return f"puzzle-{idx}"


class _FakeClient:
    """Offline generation/evaluation adapter that records calls and usage."""

    def __init__(self, generations, *, value_output="1.0", value_outputs=None,
                 generation_delay=0.0, tokens_per_call=None):
        self.generations = list(generations)
        self.value_output = value_output
        self.value_outputs = list(value_outputs) if value_outputs is not None else None
        self.generation_delay = generation_delay
        self.tokens_per_call = tokens_per_call
        self._lock = threading.Lock()
        self.generate_calls = 0
        self.value_calls = 0
        self.prompts: list[str] = []

    def __call__(self, args, prompt, n=1, stop=None, **kwargs):
        with self._lock:
            self.prompts.append(prompt)
        if self.tokens_per_call is not None:
            get_usage_tracker().record("fake-model", self.tokens_per_call, 0)
        if prompt == "[VALUE]":
            with self._lock:
                index = self.value_calls
                self.value_calls += 1
            if self.value_outputs is not None:
                return [self.value_outputs[min(index, len(self.value_outputs) - 1)]]
            return [self.value_output]
        with self._lock:
            index = self.generate_calls
            self.generate_calls += 1
        if self.generation_delay:
            time.sleep(self.generation_delay)
        if not self.generations:
            return []
        generation = self.generations[min(index, len(self.generations) - 1)]
        return [generation]


class _RecordingStrategy(BaseAssignStrategy):
    """Round-robin assignment that records every reward update it receives."""

    def __init__(self, eval_client="remote"):
        self._round_robin = RoundRobinStrategy(eval_client=eval_client)
        self.rewards: list[dict[str, float]] = []

    def assign(self, model_list, ys, context):
        return self._round_robin.assign(model_list, ys, context)

    def update(self, context, rewards):
        self.rewards.append(dict(rewards))


def _value_args(**overrides) -> DecisionSpec:
    defaults = dict(
        search_policy="baseline",
        method_generate="sample",
        prompt_sample="standard",
        method_evaluate="value",
        method_select="greedy",
        n_generate_sample=1,
        n_evaluate_sample=1,
        n_select_sample=2,
    )
    defaults.update(overrides)
    return DecisionSpec(**defaults)


def _dqn_args(**overrides) -> DecisionSpec:
    defaults = dict(
        search_policy="dqn",
        method_generate="sample",
        prompt_sample="standard",
        method_evaluate="value",
        method_select="greedy",
        n_generate_sample=1,
        n_evaluate_sample=1,
        n_select_sample=1,
        dqn_seed=1,
        dqn_epsilon=0.0,
    )
    defaults.update(overrides)
    return DecisionSpec(**defaults)


# ── Evaluator cache scope ─────────────────────────────────────────────


class TestEvaluatorCacheScope(unittest.TestCase):
    def test_same_solver_client_is_charged_once_for_a_repeated_prompt(self):
        task = _ValueTask()
        client = _FakeClient(["alpha", "beta"])
        solver = FederatedSolver(_value_args(), clients={"local": client})

        solver.solve(task, 0, to_print=False, solve_client="local")

        self.assertEqual(client.generate_calls, 2)
        self.assertEqual(
            client.value_calls, 1,
            "the same evaluator paid twice for an identical value prompt",
        )

    def test_diverse_search_reuses_prompts_without_leaking_across_inputs(self):
        for federated in (False, True):
            with self.subTest(federated=federated):
                client = _FakeClient(["alpha", "beta"], value_outputs=["1.0", "0.5"])
                solver = FederatedSolver(
                    _value_args(search_policy="diverse", diverse_joint_rank=False),
                    clients={"only": client})
                task = _IndexedInputTask()
                if federated:
                    solve = lambda idx: solver.federated_solve(
                        task, idx, to_print=False,
                        assign_strategy=RoundRobinStrategy(eval_client="only"))
                else:
                    solve = lambda idx: solver.solve(task, idx, to_print=False, solve_client="only")
                solve(0)
                self.assertEqual(client.value_calls, 1)
                _, trace = solve(1)
                self.assertEqual(client.value_calls, 2)
                self.assertEqual(trace["steps"][0]["values"], [0.5])

    def test_two_solvers_in_one_process_do_not_share_caches(self):
        task = _ValueTask()
        first_client = _FakeClient(["alpha", "beta"], value_output="1.0")
        second_client = _FakeClient(["alpha", "beta"], value_output="0.5")
        first = FederatedSolver(_value_args(), clients={"local": first_client})
        second = FederatedSolver(_value_args(), clients={"local": second_client})

        _, first_trace = first.solve(task, 0, to_print=False, solve_client="local")
        _, second_trace = second.solve(task, 0, to_print=False, solve_client="local")

        self.assertEqual(first_client.value_calls, 1)
        self.assertEqual(
            second_client.value_calls, 1,
            "a second solver was served from the first solver's cache",
        )
        self.assertEqual(first_trace["steps"][0]["values"], [1.0])
        self.assertEqual(second_trace["steps"][0]["values"], [0.5])

    def test_switching_evaluator_client_does_not_reuse_scores(self):
        task = _ValueTask()
        local = _FakeClient(["alpha", "beta"], value_output="1.0")
        remote = _FakeClient(["alpha", "beta"], value_output="0.5")
        solver = FederatedSolver(
            _value_args(), clients={"local": local, "remote": remote})

        solver.solve(task, 0, to_print=False, solve_client="local")
        _, trace = solver.solve(task, 0, to_print=False, solve_client="remote")

        self.assertEqual(remote.value_calls, 1)
        self.assertEqual(trace["steps"][0]["values"], [0.5])

    def test_federated_client_step_uses_the_task_value_cache(self):
        task = _ValueTask()
        client = _FakeClient(["alpha", "beta"])
        solver = FederatedSolver(_value_args(), clients={"only": client})

        solver.federated_solve(
            task, 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="only"),
        )

        self.assertEqual(client.generate_calls, 2)
        self.assertEqual(client.value_calls, 1)

    def test_dqn_ranked_candidates_use_the_task_value_cache(self):
        task = _ValueTask()
        client = _FakeClient(["alpha", "beta"])
        solver = FederatedSolver(_dqn_args(), clients={"only": client})

        solver.federated_solve(
            task, 0, to_print=False,
            assign_strategy=RoundRobinStrategy(eval_client="only"),
        )

        self.assertEqual(
            client.value_calls, 1,
            "DQN fallback evaluation was not served by the task value cache",
        )

    def test_reusing_a_task_object_for_a_different_input_resets_the_cache(self):
        task = _IndexedInputTask()
        client = _FakeClient(["alpha", "beta"], value_outputs=["1.0", "0.5"])
        solver = FederatedSolver(_value_args(), clients={"local": client})

        solver.solve(task, 0, to_print=False, solve_client="local")
        _, trace = solver.solve(task, 1, to_print=False, solve_client="local")

        self.assertEqual(
            client.value_calls, 2,
            "a prompt-only hit leaked one task input's score into another",
        )
        self.assertEqual(trace["steps"][0]["values"], [0.5])


# ── DQN step resources ────────────────────────────────────────────────


class TestDqnStepResources(unittest.TestCase):
    def test_parallel_generation_uses_wall_clock_and_scoped_usage_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "transitions.jsonl")
            args = _dqn_args(dqn_jsonl=path, n_select_sample=2)
            clients = {
                "local": _FakeClient(["a"], generation_delay=0.3, tokens_per_call=50),
                "remote": _FakeClient(["b"], generation_delay=0.3, tokens_per_call=50),
                "third": _FakeClient(["c"], generation_delay=0.3, tokens_per_call=50),
            }
            solver = FederatedSolver(args, clients=clients)
            global_tokens_before = get_usage_tracker().get_total_tokens()

            with usage_scope() as tracker:
                outcome = finish_task(
                    solver, _OneStepTask(), 0, method="federated_solve", to_print=False,
                    assign_strategy="round_robin", solve_client="local",
                )
                scoped_tokens = tracker.get_total_tokens()

            metrics = outcome.trace["steps"][0]["search_metrics"]
            # Three parallel generation calls plus one evaluation call, all
            # recorded into the run-scoped tracker through copied contexts.
            self.assertEqual(scoped_tokens, 200)
            self.assertEqual(metrics["step_tokens"], 150)
            self.assertEqual(metrics["eval_tokens"], 50)
            self.assertEqual(metrics["total_step_tokens"], 200)
            self.assertEqual(solver.dqn_session._cumulative_tokens, 200)
            self.assertEqual(solver.dqn_session._episode.last_terminal_costs[0], 200)
            self.assertGreaterEqual(metrics["step_latency"], 0.25)
            self.assertLess(
                metrics["step_latency"], 0.7,
                "step latency looks like the sum of parallel client durations",
            )
            self.assertEqual(
                get_usage_tracker().get_total_tokens(), global_tokens_before,
                "worker threads leaked usage into the process-global tracker",
            )
            with open(path, encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["done"])


# ── DQN source attribution ────────────────────────────────────────────


class TestDqnSourceAttribution(unittest.TestCase):
    def test_unscored_and_duplicate_candidates_are_mapped_explicitly(self):
        task = _ValueTask()
        infos = [
            {"output_ys": ["s0", "s1", "s2"]},
            {"output_ys": ["s0", "s3"]},
        ]

        # The session deduplicated and scored only s0/s2; s1/s3 were cut.
        attach_dqn_scores(task, "puzzle", ["s0", "s2"], [0.7, 0.2], infos)

        self.assertEqual(infos[0]["scored_ys"], ["s0", "s2"])
        self.assertEqual(infos[0]["values"], [0.7, 0.2])
        # A duplicate proposal from another client shares the one score...
        self.assertEqual(infos[1]["scored_ys"], ["s0"])
        self.assertEqual(infos[1]["values"], [0.7])
        # ...and the truncated candidates are absent, not 0.0 scores.
        self.assertNotIn("s1", infos[0]["scored_ys"])
        self.assertNotIn("s3", infos[1]["scored_ys"])

    def test_duplicate_proposals_reach_strategy_with_the_shared_score(self):
        task = _ValueTask()
        # Both clients propose the same state from the root; only one client
        # is assigned the second-step parent.
        local = _FakeClient(["shared", "tail"])
        remote = _FakeClient(["shared"])
        solver = FederatedSolver(
            _dqn_args(), clients={"local": local, "remote": remote})
        strategy = _RecordingStrategy(eval_client="remote")

        outcome = finish_task(
            solver, task, 0, method="federated_solve", to_print=False,
            assign_strategy=strategy, solve_client="local",
        )

        first_step = outcome.trace["steps"][0]
        infos = {info["solve_client_name"]: info for info in first_step["client_infos"]}
        self.assertEqual(set(infos), {"local", "remote"})
        for info in infos.values():
            self.assertEqual(info["scored_ys"], ["shared"])
            self.assertEqual(info["values"], [1.0])
        self.assertEqual(strategy.rewards[0], {"local": 1.0, "remote": 1.0})
        # Every reward the strategy saw came from an actually scored client.
        for step, rewards in zip(outcome.trace["steps"], strategy.rewards):
            expected = {
                info["solve_client_name"]: sum(info["values"]) / len(info["values"])
                for info in step["client_infos"] if info["values"]
            }
            self.assertEqual(rewards, expected)


# ── Empty candidate branch ────────────────────────────────────────────


class TestEmptyCandidateBranch(unittest.TestCase):
    def test_dqn_with_no_generated_candidates_has_explicit_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "transitions.jsonl")
            client = _FakeClient([])
            solver = FederatedSolver(_dqn_args(dqn_jsonl=path), clients={"only": client})

            outcome = finish_task(
                solver, _OneStepTask(), 0, method="federated_solve", to_print=False,
                assign_strategy="round_robin", solve_client="only",
            )

            self.assertEqual(len(outcome.trace["steps"]), 1)
            step = outcome.trace["steps"][0]
            self.assertEqual(step["new_ys"], [""])
            self.assertEqual(step["select_new_ys"], [""])
            self.assertEqual(step["dqn_transitions"], [])
            self.assertFalse(os.path.exists(path))


# ── Final outcome snapshot ────────────────────────────────────────────


class TestFinalOutcomeSnapshot(unittest.TestCase):
    def test_terminal_reward_and_verdict_are_final_before_return(self):
        with tempfile.TemporaryDirectory() as directory:
            path = os.path.join(directory, "transitions.jsonl")
            # Second generated state contains "win": the DQN session records a
            # provisional success, while the exact verifier returns r=0.
            client = _FakeClient(["keep", "win"])
            solver = FederatedSolver(_dqn_args(dqn_jsonl=path), clients={"only": client})

            outcome = finish_task(
                solver, _ValueTask(), 0, method="federated_solve", to_print=False,
                assign_strategy="round_robin", solve_client="only",
            )

            self.assertEqual(outcome.verdicts, ({"r": 0},))
            self.assertEqual(outcome.candidates, ("keepwin",))
            terminal = outcome.trace["steps"][-1]["dqn_transitions"][-1]
            self.assertTrue(terminal["done"])
            self.assertLess(
                terminal["reward"], 0.0,
                "terminal reward still carries provisional success",
            )
            with open(path, encoding="utf-8") as stream:
                records = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(len(records), 1)
            self.assertTrue(records[0]["done"])
            self.assertEqual(records[0]["reward"], terminal["reward"])

            snapshot = deepcopy(outcome.trace)
            # A later finalize must not rewrite the already published terminal
            # record, and must not mutate the outcome the caller holds.
            solver.dqn_session.finalize(True)
            self.assertEqual(outcome.trace, snapshot)
            with open(path, encoding="utf-8") as stream:
                after = [json.loads(line) for line in stream if line.strip()]
            self.assertEqual(after, records)


if __name__ == "__main__":
    unittest.main()
