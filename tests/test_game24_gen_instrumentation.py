"""Tests for Game24 generation telemetry recorder.

Given: mocked model completions with mixed valid/invalid/empty/malformed content.
When:  get_proposals_with_check / get_proposals_without_check / get_proposals
       are called with recorder=None and recorder=Recorder().
Then:  candidate lists are byte-identical regardless of recorder presence;
       the recorder captures raw text, parse outcomes, rejection reasons,
       retry ordinals, and aggregate counts.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

sys.path.insert(0, "src")

from tests._game24_instr_fixtures import ArgsStub, make_game24_task_stub
from federated_mcts.evaluation.game24_instrumentation import GenerationRecorder


class TestGenerationRecorderIdentity(unittest.TestCase):
    """Candidate lists must be identical with and without recorder."""

    def setUp(self):
        self.task = make_game24_task_stub()
        self.args = ArgsStub()
        self.recorder = GenerationRecorder()

    def test_candidates_identical_recorder_vs_none_with_check(self):
        """When a recorder is supplied, get_proposals_with_check returns
        the same candidate list as without."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["validA\n\nbadformatX\nvalidB"])

        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            no_rec = get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
            )

        mock_gpt2 = MagicMock(return_value=["validA\n\nbadformatX\nvalidB"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt2):
            with_rec = get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=self.recorder,
            )

        self.assertEqual(no_rec, with_rec,
            "Candidate list differs when recorder is present")

    def test_recorder_captures_rejected_proposals(self):
        """Recorder must capture empty and malformed lines with correct reasons."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["validA\n\nbadformatX\nvalidB"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=self.recorder,
            )

        self.assertEqual(len(self.recorder.steps), 1)
        step = self.recorder.steps[0]
        proposals = step["proposals"]
        reasons = [p["reason"] for p in proposals]
        self.assertIn("empty_content", reasons)
        self.assertIn("malformed_format", reasons)
        self.assertIn("accepted", reasons)

    def test_recorder_captures_retry_ordinal(self):
        """Recorder must capture retry ordinals when multiple API calls happen."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock()
        mock_gpt.side_effect = [
            ["p1\np2\np3\np4"],
            ["p5\np6"],
        ]
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=self.recorder,
            )

        step = self.recorder.steps[0]
        for p in step["proposals"]:
            self.assertEqual(p["retry_ordinal"], 0)

    def test_recorder_captures_raw_completions(self):
        """Recorder must store the raw completion strings from the model."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["lineA\nlineB\nlineC\nlineD"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=self.recorder,
            )

        step = self.recorder.steps[0]
        self.assertEqual(step["raw_completions"], ["lineA\nlineB\nlineC\nlineD"])

    def test_recorder_captures_depth_and_parent(self):
        """Recorder must capture the step depth and parent state."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["valid"])
        parent = "1 + 2 = 3 (left: 3 3 4)\n"
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(
                self.args, 2, self.task, "1 2 3 4", parent,
                recorder=self.recorder,
            )

        step = self.recorder.steps[0]
        self.assertEqual(step["depth"], 2)
        self.assertEqual(step["parent_state"], parent)

    def test_without_check_recorder(self):
        """get_proposals_without_check with recorder captures records."""
        from federated_mcts.core.generation import get_proposals_without_check

        rec = GenerationRecorder()
        mock_gpt = MagicMock(return_value=["lineA\nlineB"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            result = get_proposals_without_check(
                self.args, 1, self.task, "1 2 3 4", "parent\n",
                recorder=rec,
            )

        self.assertEqual(len(rec.steps), 1)
        step = rec.steps[0]
        self.assertEqual(len(step["proposals"]), 2)
        for p in step["proposals"]:
            self.assertTrue(p["is_correct"])
            self.assertEqual(p["reason"], "accepted")
        self.assertTrue(result[0].startswith("parent\n"))

    def test_get_proposals_passes_recorder_through(self):
        """get_proposals must forward recorder to the implementation."""
        from federated_mcts.core.generation import get_proposals

        rec = GenerationRecorder()
        mock_gpt = MagicMock(return_value=["valid"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=rec,
            )
        self.assertGreater(len(rec.steps), 0)

    def test_total_proposed_and_accepted(self):
        """Recorder aggregates must be correct.

        With 3 non-empty lines (validA, badformatX, validB) and 2 accepted,
        the retry loop collects them twice before hitting time_constraint=2.
        Each iteration records all 4 lines (including empty), so
        total_proposed=8 (4×2). total_accepted counts is_correct=True items.
        """
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["validA\n\nbadformatX\nvalidB"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(
                self.args, 0, self.task, "1 2 3 4", "",
                recorder=self.recorder,
            )

        self.assertEqual(self.recorder.total_proposed, 8)
        self.assertEqual(self.recorder.total_accepted, 4)


class TestRecorderSelectionInvariance(unittest.TestCase):
    """Selection inputs from generation+evaluation must be identical
    with and without recorders."""

    def setUp(self):
        self.args = ArgsStub()

    def test_full_pipeline_invariance(self):
        """Generate and evaluate with/without recorders: candidates and
        votes must be byte-identical."""
        from federated_mcts.core.generation import get_proposals_with_check
        from federated_mcts.core.evaluation import get_votes
        from tests._game24_instr_fixtures import make_vote_task_stub
        from federated_mcts.evaluation.game24_instrumentation import VoteRecorder

        gen_task = make_game24_task_stub()
        vote_task = make_vote_task_stub()

        gen_rec = GenerationRecorder()
        vote_rec = VoteRecorder()

        completions = ["a\nb\nc\nd"]

        gen_gpt1 = MagicMock(return_value=completions)
        with patch("federated_mcts.core.generation.gpt", gen_gpt1):
            no_rec_ys = get_proposals_with_check(
                self.args, 0, gen_task, "1 2 3 4", "",
            )

        vote_gpt1 = MagicMock(return_value=[
            "The best choice is 0",
            "The best choice is 2",
            "The best choice is 1",
        ])
        with patch("federated_mcts.core.evaluation.gpt", vote_gpt1):
            no_rec_votes = get_votes(self.args, vote_task, "x", no_rec_ys, 3)

        gen_gpt2 = MagicMock(return_value=completions)
        with patch("federated_mcts.core.generation.gpt", gen_gpt2):
            with_rec_ys = get_proposals_with_check(
                self.args, 0, gen_task, "1 2 3 4", "",
                recorder=gen_rec,
            )

        vote_gpt2 = MagicMock(return_value=[
            "The best choice is 0",
            "The best choice is 2",
            "The best choice is 1",
        ])
        with patch("federated_mcts.core.evaluation.gpt", vote_gpt2):
            with_rec_votes = get_votes(
                self.args, vote_task, "x", with_rec_ys, 3,
                recorder=vote_rec, evaluator_mode="vote",
                evaluator_identity="remote",
            )

        self.assertEqual(no_rec_ys, with_rec_ys)
        self.assertEqual(no_rec_votes, with_rec_votes)
        self.assertEqual(gen_rec.total_proposed, 4)
        self.assertEqual(gen_rec.total_accepted, 4)
        self.assertEqual(len(vote_rec.steps), 1)
        self.assertEqual(vote_rec.steps[0]["vote_totals"], [1, 1, 1, 0])


if __name__ == "__main__":
    unittest.main()
