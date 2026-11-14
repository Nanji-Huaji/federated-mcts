"""JSON safety tests for Game24 candidate-flow metrics.

All metrics structures must survive json.dumps(allow_nan=False).
"""

import json
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_metrics import (
    make_candidate_event,
    make_step_metrics,
    make_task_metrics,
    summarize_experiment_metrics,
)


class TestJsonSafety(unittest.TestCase):
    """Given any valid metrics, When json.dumps(allow_nan=False) is called,
       Then it succeeds without raising ValueError."""

    def test_task_is_json_safe(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(
                depth=0, parent_count=1, candidates=[
                    make_candidate_event(task_id="t", depth=0,
                                         generation_ordinal=0,
                                         stage="terminal_solved",
                                         disposition="terminal_solved",
                                         is_terminal=True,
                                         evaluator_score=1.0),
                ])])
        result = json.dumps(task, allow_nan=False)
        self.assertIsInstance(result, str)
        self.assertIn("terminal_solved", result)

    def test_experiment_is_json_safe(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(depth=0, parent_count=1,
                                            candidates=[])])
        exp = summarize_experiment_metrics([task])
        result = json.dumps(exp, allow_nan=False)
        self.assertIsInstance(result, str)
        self.assertIn("version", result)

    def test_experiment_empty_is_json_safe(self):
        exp = summarize_experiment_metrics([])
        result = json.dumps(exp, allow_nan=False)
        self.assertIsInstance(result, str)

    def test_rejection_counts_json_safe(self):
        task = make_task_metrics(
            task_id="t", task_input="1 2 3 4",
            step_details=[make_step_metrics(depth=0, parent_count=1, candidates=[
                make_candidate_event(
                    task_id="t", depth=0, generation_ordinal=0,
                    stage="proposed", rejection_reason="parse_error"),
                make_candidate_event(
                    task_id="t", depth=0, generation_ordinal=1,
                    stage="proposed", rejection_reason="vote_parse_failed"),
            ])])
        result = json.dumps(task, allow_nan=False)
        parsed = json.loads(result)
        self.assertEqual(parsed["rejection_counts"]["parse_error"], 1)
        self.assertEqual(
            parsed["rejection_counts"]["vote_parse_failed"], 1)
        self.assertEqual(
            parsed["rejection_counts"].get("empty_content", 0), 0)


if __name__ == "__main__":
    unittest.main()
