"""Regression tests for get_proposals generation budget and call-count defects.

PRODUCTION DEFECT: get_proposals_with_check hardcodes n=1 per API call,
loops up to 6 calls to collect 4 proposals, and only reads the first
completion [0] from each response.  Expected contract: request a budget
of n from the client, accept valid lines from all completions in each
response, deduplicate, and make at most 2 calls when the first response
is sufficient.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock

sys.path.insert(0, "src")


class ArgsStub:
    check_format = True


def _make_task_stub():
    task = MagicMock()
    task.pre_generate_check.return_value = True
    task.propose_prompt_wrap.return_value = "mock prompt"
    task.process_generate_result.side_effect = lambda pro, x, y, check: (True, pro.strip())
    return task


class TestGetProposalsBudgetHonoured(unittest.TestCase):

    def setUp(self):
        self.task = _make_task_stub()
        self.args = ArgsStub()

    def test_client_receives_n_equal_to_generation_budget(self):
        """When get_proposals_with_check is called, the client should receive
        n equal to the generation budget (default: 4), not hardcoded 1."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["a\nb\nc\nd"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            get_proposals_with_check(self.args, 0, self.task, "x", "y", client=None)

        call_kwargs = mock_gpt.call_args.kwargs
        self.assertEqual(
            call_kwargs.get("n"), 4,
            "DEFECT: n is hardcoded to 1.  Expected: n=4 (generation budget).",
        )

    def test_all_completions_accepted_not_just_first(self):
        """When the API returns multiple completions (one per requested n),
        all should be accepted, not just the first [0]."""
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock(return_value=["comp-A", "comp-B", "comp-C", "comp-D"])
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            result = get_proposals_with_check(self.args, 0, self.task, "x", "y", client=None)

        self.assertIn("comp-A", result)
        self.assertIn(
            "comp-B", result,
            "DEFECT: only [0] is read; comp-B is discarded.",
        )
        self.assertIn("comp-C", result)
        self.assertIn("comp-D", result)

    def test_proposals_are_deduplicated(self):
        from federated_mcts.core.generation import get_proposals_with_check

        mock_gpt = MagicMock()
        mock_gpt.side_effect = [
            ["dup\nunique-1\ndup"],
            ["unique-2"],
            ["dup\nunique-1\ndup"],
            ["unique-2"],
            ["dup"],
            ["unique-3"],
        ]
        with patch("federated_mcts.core.generation.gpt", mock_gpt):
            result = get_proposals_with_check(self.args, 0, self.task, "x", "y", client=None)

        self.assertEqual(len(result), len(set(result)))

    def test_at_most_two_calls_when_budget_is_honoured(self):
        """When each completion yields one proposal, a budget of n=4 allows
        one call; even with some invalid proposals, at most 2 calls should
        suffice.  Currently, n=1 forces >= 4 calls."""
        from federated_mcts.core.generation import get_proposals_with_check

        call_count = 0

        def one_per_call(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return ["p"]  # one valid proposal per completion

        self.task.process_generate_result.side_effect = lambda p, x, y, c: (bool(p.strip()), p.strip())
        with patch("federated_mcts.core.generation.gpt", side_effect=one_per_call):
            get_proposals_with_check(self.args, 0, self.task, "x", "y", client=None)

        self.assertLessEqual(
            call_count, 2,
            "DEFECT: n=1 forces 4+ calls to collect 4 proposals.  "
            f"Actual calls: {call_count}.  Expected: <= 2 with n=4 budget.",
        )


if __name__ == "__main__":
    unittest.main()
