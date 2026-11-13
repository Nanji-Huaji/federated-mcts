"""Contract tests for the core search-observation seam and evaluator cache.

Given: the real ``Game24Task`` and a scripted fake model client.
When:  generation/vote observation is enabled, disabled, or routed through a
       no-op sink, and two evaluators score the same prompt.
Then:  candidates, scores and client call counts are unchanged by observation;
       the shared vote parser is the only parser; core code loads no Game24
       evaluation modules; and cached value scores never cross evaluator
       identities.
"""

from __future__ import annotations

import os
import subprocess
import sys
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from federated_mcts.core.evaluation import (
    EvaluatorValueCache,
    get_values,
    get_votes,
)
from federated_mcts.core.generation import (
    get_proposals,
    get_proposals_with_check,
    get_proposals_without_check,
)
from federated_mcts.core.observation import (
    NULL_OBSERVATION_SINK,
    VOTE_CHOICE_PATTERN,
    parse_vote_choice,
)
from federated_mcts.evaluation.game24_instrumentation import (
    GenerationRecorder,
    VoteRecorder,
    parse_vote_response,
)
from federated_mcts.tasks.game24 import Game24Task


class _Args:
    check_format = True
    eval_rule = False
    n_generate_sample = 4
    n_evaluate_sample = 2


class _ScriptedClient:
    """Fake model client: returns scripted completions and counts calls."""

    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0
        self.prompts: list[str] = []

    def __call__(self, args, prompt, n=1, stop=None, get_logprobs=False):
        self.prompts.append(prompt)
        response = self.responses[min(self.calls, len(self.responses) - 1)]
        self.calls += 1
        return response


def _generation_client():
    return _ScriptedClient([
        ["1 + 2 = 3 (left: 3 3 4)", "1 * 2 = 2 (left: 2 3 4)"],
    ])


def _vote_client():
    return _ScriptedClient([[
        "The best choice is 0",
        "The best choice is 1",
    ]])


class _EvaluationTask:
    """Minimal real-shaped task for the value path (Game24 task stays real)."""

    value_cache: dict = {}

    @staticmethod
    def value_prompt_wrap(x, y):
        return f"value::{x}::{y}"

    @staticmethod
    def value_outputs_unwrap(x, y, outputs):
        mapping = {"impossible": 0, "likely": 1, "sure": 2}
        return sum(mapping.get(out.split("\n")[-1], 0) for out in outputs)

    @staticmethod
    def pre_value_check(y, eval_rule):
        return 0, False

    @staticmethod
    def canonical_state_key(x, y):
        return (x, y)


class TestGenerationObservationIdentity(unittest.TestCase):
    """Recorder presence must not change candidates or call counts."""

    def setUp(self):
        self.task = Game24Task()
        self.args = _Args()
        self.parent = ""
        self.x = "1 2 3 4"

    def _run(self, **kwargs):
        client = _generation_client()
        candidates = get_proposals(
            self.args, 0, self.task, self.x, self.parent,
            client=client, **kwargs,
        )
        return candidates, client.calls

    def test_recorder_does_not_change_candidates_or_calls(self):
        without, calls_without = self._run()
        recorder = GenerationRecorder()
        with_recorder, calls_with = self._run(recorder=recorder)

        self.assertEqual(without, with_recorder)
        self.assertEqual(calls_without, calls_with)
        self.assertEqual(len(recorder.steps), 1)

    def test_noop_sink_matches_uninstrumented_run(self):
        without, calls_without = self._run()
        with_noop, calls_noop = self._run(recorder=NULL_OBSERVATION_SINK)

        self.assertEqual(without, with_noop)
        self.assertEqual(calls_without, calls_noop)

    def test_unchecked_path_recorder_identity(self):
        self.args.check_format = False
        try:
            without, calls_without = self._run()
            recorder = GenerationRecorder()
            with_recorder, calls_with = self._run(recorder=recorder)
        finally:
            self.args.check_format = True

        self.assertEqual(without, with_recorder)
        self.assertEqual(calls_without, calls_with)
        self.assertEqual(recorder.steps[0]["retry_count"], 1)

    def test_records_keep_baseline_telemetry_fields(self):
        recorder = GenerationRecorder()
        get_proposals_with_check(
            self.args, 0, self.task, self.x, self.parent,
            client=_generation_client(), recorder=recorder,
        )
        step = recorder.steps[0]

        self.assertEqual(
            set(step),
            {"depth", "parent_index", "parent_state", "raw_completions",
             "retry_count", "proposals"},
        )
        self.assertEqual(
            set(step["proposals"][0]),
            {"depth", "parent_index", "retry_ordinal", "completion_index",
             "raw_line", "is_correct", "updated_text", "reason"},
        )
        self.assertEqual(step["retry_count"], 2)
        self.assertEqual(
            step["raw_completions"],
            ["1 + 2 = 3 (left: 3 3 4)", "1 * 2 = 2 (left: 2 3 4)"] * 2,
        )

    def test_unchecked_records_keep_baseline_fields(self):
        recorder = GenerationRecorder()
        self.args.check_format = False
        try:
            get_proposals_without_check(
                self.args, 0, self.task, self.x, self.parent,
                client=_generation_client(), recorder=recorder,
            )
        finally:
            self.args.check_format = True
        self.assertEqual(recorder.steps[0]["proposals"][0]["reason"], "accepted")
        self.assertTrue(recorder.steps[0]["proposals"][0]["is_correct"])


class TestVoteObservationIdentity(unittest.TestCase):
    """Vote recorder presence must not change totals or call counts."""

    def setUp(self):
        self.task = Game24Task()
        self.args = _Args()
        self.ys = [
            "1 + 2 = 3 (left: 3 3 4)\n",
            "3 * 4 = 12 (left: 12)\n",
            "2 * 12 = 24 (left: 24)\n",
        ]

    def _run(self, **kwargs):
        client = _vote_client()
        values = get_votes(
            self.args, self.task, "1 2 3 4", self.ys, 2,
            client=client, evaluator_mode="vote",
            evaluator_identity="fake-client", depth=1, **kwargs,
        )
        return values, client.calls

    def test_recorder_does_not_change_votes_or_calls(self):
        without, calls_without = self._run()
        recorder = VoteRecorder()
        with_recorder, calls_with = self._run(recorder=recorder)

        self.assertEqual(without, with_recorder)
        self.assertEqual(calls_without, calls_with)
        self.assertEqual(recorder.steps[0]["vote_totals"], without)

    def test_noop_sink_matches_uninstrumented_run(self):
        without, calls_without = self._run()
        with_noop, calls_noop = self._run(recorder=NULL_OBSERVATION_SINK)
        self.assertEqual(without, with_noop)
        self.assertEqual(calls_without, calls_noop)

    def test_records_keep_baseline_telemetry_fields(self):
        recorder = VoteRecorder()
        self._run(recorder=recorder)
        step = recorder.steps[0]
        self.assertEqual(
            set(step),
            {"depth", "n_candidates", "evaluator_mode", "evaluator_identity",
             "call_count", "vote_totals", "calls"},
        )
        self.assertEqual(
            set(step["calls"][0]),
            {"call_index", "evaluator_mode", "evaluator_identity", "raw_output",
             "parsed_index", "parse_success", "failure_reason", "in_range"},
        )
        self.assertEqual(step["evaluator_mode"], "vote")
        self.assertEqual(step["evaluator_identity"], "fake-client")
        self.assertEqual(step["depth"], 1)

    def test_out_of_range_vote_flagged_by_real_task(self):
        recorder = VoteRecorder()
        client = _ScriptedClient([["The best choice is 9"]])
        values = get_votes(
            self.args, self.task, "1 2 3 4", self.ys, 1, client=client,
            recorder=recorder, evaluator_mode="vote",
            evaluator_identity="fake-client",
        )
        self.assertEqual(values, [0, 0, 0])
        call = recorder.steps[0]["calls"][0]
        self.assertTrue(call["parse_success"])
        self.assertFalse(call["in_range"])
        self.assertEqual(call["failure_reason"], "out_of_range")


class TestSingleVoteParser(unittest.TestCase):
    """Decision path and telemetry share exactly one parser."""

    BATTERY = [
        "The best choice is 0",
        "the BEST choice is 2\ntrailing",
        "The best choice is -1",
        "The best choice is True",
        "The best choice is",
        "",
        "choice: NaN",
        "The best choice is 42",
    ]

    def test_parse_vote_response_delegates_to_core_parser(self):
        for text in self.BATTERY:
            choice = parse_vote_choice(text)
            self.assertEqual(
                parse_vote_response(text),
                (choice.vote_index, choice.matched, choice.failure_reason),
                text,
            )

    def test_task_decision_matches_shared_parser(self):
        for text in self.BATTERY:
            choice = parse_vote_choice(text)
            expected = [0] * 3
            if choice.matched and choice.vote_index is not None and 0 <= choice.vote_index < 3:
                expected[choice.vote_index] = 1
            self.assertEqual(
                Game24Task.vote_outputs_unwrap([text], 3), expected, text)

    def test_instrumentation_pattern_is_the_core_pattern(self):
        from federated_mcts.evaluation.game24_instrumentation import (
            _VOTE_PARSE_PATTERN,
        )
        self.assertIs(_VOTE_PARSE_PATTERN, VOTE_CHOICE_PATTERN)


class TestEvaluatorValueCacheIsolation(unittest.TestCase):
    """Cached value scores are scoped by evaluator identity."""

    def setUp(self):
        self.args = _Args()
        self.task = _EvaluationTask()
        self.y = "3 3 4 (left: 3 3 4)\n"

    def _one_value(self, client, **kwargs):
        return get_values(
            self.args, self.task, "1 2 3 4", [self.y], 1, client=client, **kwargs,
        )[0]

    def test_two_evaluators_same_prompt_do_not_share_scores(self):
        first = _ScriptedClient([["likely"]])
        second = _ScriptedClient([["sure"]])

        self.assertEqual(self._one_value(first), 1)
        self.assertEqual(self._one_value(second), 2)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)
        self.assertEqual(self.task.value_cache, {})

    def test_shared_cache_separates_evaluator_identities(self):
        cache = EvaluatorValueCache()
        first = _ScriptedClient([["likely"]])
        second = _ScriptedClient([["sure"]])

        self.assertEqual(
            self._one_value(first, cache=cache, evaluator_identity="alpha"), 1)
        self.assertEqual(
            self._one_value(second, cache=cache, evaluator_identity="beta"), 2)
        self.assertEqual(len(cache), 2)
        self.assertEqual(first.calls, 1)
        self.assertEqual(second.calls, 1)

    def test_same_identity_hits_cache(self):
        cache = EvaluatorValueCache()
        client = _ScriptedClient([["sure"]])

        self.assertEqual(
            self._one_value(client, cache=cache, evaluator_identity="alpha"), 2)
        self.assertEqual(first_calls := client.calls, 1)
        self.assertEqual(
            self._one_value(client, cache=cache, evaluator_identity="alpha"), 2)
        self.assertEqual(client.calls, first_calls)


class TestCoreObservationDependencyDirection(unittest.TestCase):
    """Core search modules must not import Game24 evaluation modules."""

    def test_core_imports_load_no_evaluation_modules(self):
        repo_root = Path(__file__).resolve().parents[1]
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(repo_root / "src"), env.get("PYTHONPATH", "")]).strip(os.pathsep)
        code = (
            "import sys\n"
            "import federated_mcts.core.observation\n"
            "import federated_mcts.core.generation\n"
            "import federated_mcts.core.evaluation\n"
            "import federated_mcts.core.joint_ranking\n"
            "print([m for m in sys.modules"
            " if m.startswith('federated_mcts.evaluation')])\n"
        )
        completed = subprocess.run(
            [sys.executable, "-c", code], cwd=str(repo_root), env=env,
            capture_output=True, text=True, timeout=120,
        )
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "[]", completed.stderr)


if __name__ == "__main__":
    unittest.main()
