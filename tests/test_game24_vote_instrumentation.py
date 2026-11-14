"""Tests for Game24 vote telemetry recorder.

Given: mocked vote outputs with mixed valid/invalid/out-of-range/no-match content.
When:  get_votes is called with recorder=None and recorder=Recorder().
Then:  vote totals are byte-identical regardless of recorder presence;
       the recorder captures raw text, parsed indices, failure reasons,
       evaluator identity, and parse failure counts.
"""

import re
import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from tests._game24_instr_fixtures import ArgsStub, make_vote_task_stub
from federated_mcts.evaluation.game24_instrumentation import (
    VoteRecorder,
    parse_vote_response,
)


class TestVoteRecorderIdentity(unittest.TestCase):
    """Vote values must be identical with and without recorder."""

    def setUp(self):
        self.task = make_vote_task_stub()
        self.args = ArgsStub()

    def test_votes_identical_recorder_vs_none(self):
        """Vote totals must not change when recorder is present."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        ys = ["cand0", "cand1", "cand2"]
        mock_gpt = MagicMock(return_value=[
            "The best choice is 0",
            "The best choice is 2",
            "The best choice is 0",
        ])

        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            no_rec = get_votes(self.args, self.task, "x", ys, 3)

        mock_gpt2 = MagicMock(return_value=[
            "The best choice is 0",
            "The best choice is 2",
            "The best choice is 0",
        ])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt2):
            with_rec = get_votes(self.args, self.task, "x", ys, 3,
                                 recorder=recorder, evaluator_mode="vote",
                                 evaluator_identity="remote")

        self.assertEqual(no_rec, with_rec)

    def test_recorder_captures_parsed_votes(self):
        """Recorder must capture each vote's parsed index."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        ys = ["c0", "c1", "c2"]
        mock_gpt = MagicMock(return_value=[
            "The best choice is 0",
            "The best choice is 2",
            "The best choice is 0",
        ])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, self.task, "x", ys, 3,
                      recorder=recorder, evaluator_mode="vote",
                      evaluator_identity="remote")

        self.assertEqual(len(recorder.steps), 1)
        step = recorder.steps[0]
        self.assertEqual(step["vote_totals"], [2, 0, 1])
        self.assertEqual(step["call_count"], 3)
        self.assertEqual(step["n_candidates"], 3)

        parsed = [c["parsed_index"] for c in step["calls"]]
        self.assertEqual(parsed, [0, 2, 0])

    def test_recorder_captures_parse_failures(self):
        """Recorder must capture no-match and out-of-range votes."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        ys = ["c0", "c1"]
        mock_gpt = MagicMock(return_value=[
            "nonsense with no index",
            "The best choice is 5",
            "The best choice is 0",
        ])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, self.task, "x", ys, 3,
                      recorder=recorder, evaluator_mode="vote",
                      evaluator_identity="remote")

        step = recorder.steps[0]
        calls = step["calls"]

        self.assertFalse(calls[0]["parse_success"])
        self.assertEqual(calls[0]["failure_reason"], "no_match")
        self.assertFalse(calls[0]["in_range"])

        self.assertEqual(calls[1]["parsed_index"], 5)

        self.assertTrue(calls[2]["parse_success"])
        self.assertEqual(calls[2]["parsed_index"], 0)
        self.assertTrue(calls[2]["in_range"])

        self.assertEqual(recorder.total_parse_failures, 2)

    def test_recorder_captures_evaluator_identity(self):
        """Recorder must persist evaluator_mode and evaluator_identity."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        mock_gpt = MagicMock(return_value=["The best choice is 0"])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, self.task, "x", ["c0"], 1,
                      recorder=recorder, evaluator_mode="vote",
                      evaluator_identity="qwen-72b")

        step = recorder.steps[0]
        self.assertEqual(step["evaluator_mode"], "vote")
        self.assertEqual(step["evaluator_identity"], "qwen-72b")
        for c in step["calls"]:
            self.assertEqual(c["evaluator_mode"], "vote")
            self.assertEqual(c["evaluator_identity"], "qwen-72b")

    def test_vote_totals_identical_with_failures(self):
        """Vote totals with mixed failures must match between recorder/none."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        ys = ["c0", "c1"]
        mock_gpt = MagicMock(return_value=[
            "nonsense",
            "The best choice is 0",
            "garbage",
            "The best choice is 0",
            "The best choice is 5",
        ])

        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            no_rec = get_votes(self.args, self.task, "x", ys, 5)

        mock_gpt2 = MagicMock(return_value=[
            "nonsense",
            "The best choice is 0",
            "garbage",
            "The best choice is 0",
            "The best choice is 5",
        ])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt2):
            with_rec = get_votes(self.args, self.task, "x", ys, 5,
                                 recorder=recorder, evaluator_mode="vote",
                                 evaluator_identity="test")

        self.assertEqual(no_rec, with_rec)

    def test_negative_vote_handled_correctly(self):
        """Negative parsed indices won't match the regex, so parse_success=False."""
        from federated_mcts.core.evaluation import get_votes

        recorder = VoteRecorder()
        mock_gpt = MagicMock(return_value=["The best choice is -1"])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, self.task, "x", ["c0"], 1,
                      recorder=recorder, evaluator_mode="vote",
                      evaluator_identity="test")

        step = recorder.steps[0]
        call = step["calls"][0]
        self.assertIsNone(call["parsed_index"])
        self.assertFalse(call["parse_success"])
        self.assertEqual(call["failure_reason"], "no_match")
        self.assertFalse(call["in_range"])


# ═══════════════════════════════════════════════════════════════════
#  Boundary and parser-equivalence tests
# ═══════════════════════════════════════════════════════════════════

_VOTE_PATTERN_TASK = re.compile(
    r".*best choice is.*?(?<!-)(\d+)", re.DOTALL | re.IGNORECASE
)


class TestVoteParserEquivalence(unittest.TestCase):
    """parse_vote_response mirrors Game24Task.vote_outputs_unwrap exactly."""

    def setUp(self):
        self.args = ArgsStub()

    def test_identical_regex(self):
        from federated_mcts.evaluation.game24_instrumentation import _VOTE_PARSE_PATTERN
        self.assertEqual(_VOTE_PARSE_PATTERN.pattern, _VOTE_PATTERN_TASK.pattern)
        self.assertEqual(_VOTE_PARSE_PATTERN.flags, _VOTE_PATTERN_TASK.flags)

    def test_parser_equivalence_happy(self):
        for text in [
            "The best choice is 0",
            "the BEST choice is 1",
            "Choice 0 does X.\nThe best choice is 1",
            "The best choice is 2\n\nanalysis",
        ]:
            idx_t, ok_t, _ = parse_vote_response(text)
            match_t = _VOTE_PATTERN_TASK.search(text)
            self.assertEqual(ok_t, match_t is not None, f"Mismatch: {text!r}")
            if match_t:
                self.assertEqual(idx_t, int(match_t.group(1)))

    def test_parser_equivalence_no_match(self):
        for text in [
            "nonsense with no index",
            "The best choice is -1",
            "The best choice is -5",
            "best choice is True",
            "choice: NaN",
            "",
            "the best choice is",
        ]:
            _, ok_t, _ = parse_vote_response(text)
            match_t = _VOTE_PATTERN_TASK.search(text)
            self.assertEqual(ok_t, match_t is not None, f"Mismatch: {text!r}")

    def test_numeric_edge_cases(self):
        for text, expected in [
            ("The best choice is 42", 42),
            ("The best choice is 0", 0),
            ("The best choice is 10", 10),
        ]:
            idx, ok, _ = parse_vote_response(text)
            self.assertTrue(ok)
            self.assertEqual(idx, expected)

    def test_provider_n1_response(self):
        from federated_mcts.core.evaluation import get_votes
        mock_gpt = MagicMock(return_value=["The best choice is 0"])
        rec = VoteRecorder()
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, make_vote_task_stub(), "x", ["c0"], 1,
                      recorder=rec, evaluator_mode="vote",
                      evaluator_identity="deepseek-n1")
        self.assertEqual(rec.steps[0]["call_count"], 1)

    def test_provider_tuple_response_unwrapped(self):
        from federated_mcts.core.evaluation import get_votes
        mock_gpt = MagicMock(return_value=(
            ["The best choice is 2", "The best choice is 1"], [0.5, 0.8]))
        rec = VoteRecorder()
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            result = get_votes(self.args, make_vote_task_stub(), "x",
                               ["a", "b", "c"], 2, recorder=rec,
                               evaluator_mode="vote",
                               evaluator_identity="test")
        self.assertEqual(result, [0, 1, 1])
        self.assertEqual(rec.steps[0]["call_count"], 2)

    def test_out_of_range_index_detected(self):
        from federated_mcts.core.evaluation import get_votes
        mock_gpt = MagicMock(return_value=["The best choice is 5"])
        rec = VoteRecorder()
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            get_votes(self.args, make_vote_task_stub(), "x", ["a", "b", "c"], 1,
                      recorder=rec, evaluator_mode="vote",
                      evaluator_identity="test")
        call = rec.steps[0]["calls"][0]
        self.assertTrue(call["parse_success"])
        self.assertEqual(call["parsed_index"], 5)
        self.assertFalse(call["in_range"])
        self.assertEqual(call["failure_reason"], "out_of_range")

    def test_malformed_bool_like_index(self):
        _, ok, reason = parse_vote_response("The best choice is True")
        self.assertFalse(ok)
        self.assertEqual(reason, "no_match")

    def test_empty_or_whitespace_output(self):
        for text in ["", "   ", "\n\n"]:
            _, ok, reason = parse_vote_response(text)
            self.assertFalse(ok)
            self.assertEqual(reason, "no_match")

    def test_recorder_none_does_not_allocate_calls(self):
        from federated_mcts.core.evaluation import get_votes
        mock_gpt = MagicMock(return_value=["The best choice is 0"])
        with patch("federated_mcts.core.evaluation.gpt", mock_gpt):
            result = get_votes(self.args, make_vote_task_stub(), "x", ["c0"], 1)
        self.assertEqual(result, [1])


if __name__ == "__main__":
    unittest.main()
