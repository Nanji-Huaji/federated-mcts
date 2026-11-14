"""Exact Game24 local trajectory validation tests — core arithmetic/parser units.

Covers Todo 1 from .omo/plans/game24-model-only-improvements.md:
- Exact Fraction arithmetic for check_final_result and check_and_fix_last_line
- Non-integral division rejection (not floored)
- Exact integral division acceptance
- Divide-by-zero, number reuse, malformed input, negative result handling

Heuristic and routing/integration tests are in test_game24_heuristic.py
and test_game24_routing.py respectively.
"""

import sys
import unittest

sys.path.insert(0, "src")


class TestBaselineValidTrajectory(unittest.TestCase):
    """Characterization tests proving valid integral trajectories pass."""

    def test_check_final_result_valid_single_step(self):
        """1+2=3 with input 1 2 3 4 passes local validation."""
        from federated_mcts.utils.format_check import check_final_result
        correct, updated = check_final_result(
            "1 + 2 = 3 (left: 3 3 4)", x="1 2 3 4",
        )
        self.assertTrue(correct, "DEFECT: 1+2=3 should be locally valid")
        self.assertIn("3 3 4", updated)

    def test_check_final_result_valid_multiplication_step(self):
        """3*3=9 with remaining 4 9 passes local validation."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "3 * 3 = 9 (left: 4 9)", pre_line="1 + 2 = 3 (left: 3 3 4)",
        )
        self.assertTrue(correct, "DEFECT: 3*3=9 should be locally valid")

    def test_check_final_result_valid_subtraction_step(self):
        """10-4=6 with input 4 10 4 8 passes local validation."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "10 - 4 = 6 (left: 4 6 8)", x="4 10 4 8",
        )
        self.assertTrue(correct, "DEFECT: 10-4=6 should be locally valid")


class TestNonIntegralDivisionRejected(unittest.TestCase):
    """Regression: non-integral division must be rejected, not floored."""

    def test_check_final_result_three_div_two_rejected(self):
        """3/2=1 floors to 1; exact Fraction rejects it."""
        from federated_mcts.utils.format_check import check_final_result
        correct, updated = check_final_result(
            "3 / 2 = 1 (left: 1 4 8)", x="2 3 4 8",
        )
        self.assertFalse(correct,
                         "DEFECT: 3/2=1 must be REJECTED (non-integral quotient)")

    def test_check_final_result_non_integral_fraction_rejected(self):
        """5/3=1 floors to 1; exact Fraction rejects it."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "5 / 3 = 1 (left: 1 1 8)", x="3 5 1 8",
        )
        self.assertFalse(correct,
                         "DEFECT: 5/3=1 must be REJECTED (non-integral quotient)")

    def test_check_final_result_fraction_result_with_integer_numerator_rejected(self):
        """7/3=2 also floors to 2; exact Fraction rejects it."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "7 / 3 = 2 (left: 1 2 8)", x="3 7 1 8",
        )
        self.assertFalse(correct,
                         "DEFECT: 7/3=2 must be REJECTED (7/3 not integral)")

    def test_check_and_fix_last_line_non_integral_rejected(self):
        """check_and_fix_last_line must also reject non-integral division."""
        from federated_mcts.utils.format_check import check_and_fix_last_line
        correct, _ = check_and_fix_last_line(
            "3 / 2 = 1 (left: 1 4 8)", x="2 3 4 8",
        )
        self.assertFalse(correct,
                         "DEFECT: check_and_fix_last_line must reject 3/2=1")


class TestExactIntegralDivisionAccepted(unittest.TestCase):
    """Division with integral quotient must be accepted."""

    def test_six_div_three_accepted(self):
        """6/3=2 is exact integral division."""
        from federated_mcts.utils.format_check import check_final_result
        correct, updated = check_final_result(
            "6 / 3 = 2 (left: 2 4 8)", x="4 6 3 8",
        )
        self.assertTrue(correct, "DEFECT: 6/3=2 must be accepted as valid")
        self.assertIn("2 4 8", updated)

    def test_eight_div_four_accepted(self):
        """8/4=2 is exact integral division."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "8 / 4 = 2 (left: 2 3 6)", x="3 8 4 6",
        )
        self.assertTrue(correct, "DEFECT: 8/4=2 must be accepted as valid")

    def test_twelve_div_six_accepted(self):
        """12/6=2 is exact integral division."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "12 / 6 = 2 (left: 2 4 8)", x="4 12 6 8",
        )
        self.assertTrue(correct, "DEFECT: 12/6=2 must be accepted as valid")


class TestDivideByZeroRejected(unittest.TestCase):
    """Division by zero must always be rejected."""

    def test_check_final_result_div_by_zero_rejected(self):
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "5 / 0 = 0 (left: 0 1 8)", x="5 0 1 8",
        )
        self.assertFalse(correct, "DEFECT: division by zero must be rejected")

    def test_check_and_fix_last_line_div_by_zero_rejected(self):
        from federated_mcts.utils.format_check import check_and_fix_last_line
        correct, _ = check_and_fix_last_line(
            "5 / 0 = 0 (left: 0 1 8)", x="5 0 1 8",
        )
        self.assertFalse(correct, "DEFECT: division by zero must be rejected")


class TestNumberReuseRejected(unittest.TestCase):
    """Using numbers not in the available multiset is rejected."""

    def test_number_not_in_pre_left_num_rejected(self):
        """Using 9 when the pool is 1 2 3 4 must be rejected."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "9 + 1 = 10 (left: 10 2 3 4)", x="1 2 3 4",
        )
        self.assertFalse(correct, "DEFECT: using 9 from {1,2,3,4} must be rejected")

    def test_number_used_twice_when_only_once_available_rejected(self):
        """Using 3 twice when only one 3 available must be rejected."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "3 + 3 = 6 (left: 1 4 6)", x="1 3 4 8",
        )
        self.assertFalse(correct,
                         "DEFECT: using 3 twice from {1,3,4,8} must be rejected")

    def test_duplicate_number_with_enough_copies_accepted(self):
        """Using two 3s when two 3s are available is valid."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "3 + 3 = 6 (left: 4 6 8)", x="3 3 4 8",
        )
        self.assertTrue(correct,
                        "DEFECT: two 3s from {3,3,4,8} should be valid")


class TestMalformedInputRejected(unittest.TestCase):
    """Grossly malformed inputs must be rejected."""

    def test_empty_proposal_rejected(self):
        from federated_mcts.utils.format_check import check_and_fix_last_line
        is_correct, _ = check_and_fix_last_line("", "1 2 3 4")
        self.assertFalse(is_correct, "DEFECT: empty proposal must be rejected")

    def test_no_left_format_rejected(self):
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "1 + 2 = 3", x="1 2 3 4",
        )
        self.assertFalse(correct, "DEFECT: missing '(left: ...)' must be rejected")

    def test_non_digit_operands_rejected(self):
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "a + b = 3 (left: 3 4)", x="a b 3 4",
        )
        self.assertFalse(correct, "DEFECT: non-digit pre_left_num must be rejected")

    def test_bad_equation_format_rejected(self):
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "1 ^ 2 = 3 (left: 3 4)", x="1 2 3 4",
        )
        self.assertFalse(correct, "DEFECT: unsupported operator must be rejected")


class TestNegativeResultHandling(unittest.TestCase):
    """Subtraction producing negative results must still be locally valid."""

    def test_negative_result_accepted_if_arithmetically_correct(self):
        """1-6=-5 from {1,6,3,4} is arithmetically correct and must be accepted."""
        from federated_mcts.utils.format_check import check_final_result
        correct, updated = check_final_result(
            "1 - 6 = -5 (left: -5 3 4)", x="1 6 3 4",
        )
        self.assertTrue(correct,
                        "DEFECT: 1-6=-5 is arithmetically correct, must be valid")
        self.assertIn("-5 3 4", updated)

    def test_zero_subtraction_accepted(self):
        """5-5=0 is arithmetically correct and must be accepted."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "5 - 5 = 0 (left: 0 1 1)", x="5 5 1 1",
        )
        self.assertTrue(correct, "DEFECT: 5-5=0 is exact, must be valid")


class TestLeftMultisetOrderIndependence(unittest.TestCase):
    """The `left:` remain-set is a multiset; its printed order must not matter.

    Regression: `check_final_result` compared the model's printed `left:` list
    against an internally sorted one with `!=`, so an arithmetically correct step
    was rejected whenever the model printed the remain-set out of ascending order.
    That silently scored correct Game24 solutions as failures.
    """

    def test_unsorted_left_rejected_before_fix(self):
        """1+2=3 with remain-set {3,3,4} printed as '4 3 3' is still valid."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "1 + 2 = 3 (left: 4 3 3)", x="1 2 3 4",
        )
        self.assertTrue(
            correct,
            "DEFECT: '1 + 2 = 3' is arithmetically correct; the remaining "
            "multiset {3,3,4} must be accepted in any printed order",
        )

    def test_unsorted_left_two_element_remainder(self):
        """3+3=6 from {3,3,4,8} leaves {4,6,8}; '8 4 6' order must be accepted."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "3 + 3 = 6 (left: 8 4 6)", x="3 3 4 8",
        )
        self.assertTrue(
            correct,
            "DEFECT: remain-set {4,6,8} printed as '8 4 6' must be accepted",
        )

    def test_pilot_task_3_correct_solution_accepted(self):
        """The textbook solution to '1 1 1 8' must be scored successful.

        Captured from the B0 pilot: the model produced this exact trajectory and
        the grader returned False purely because each `left:` was unsorted.
        """
        from federated_mcts.tasks.game24 import Game24Task
        task = Game24Task()
        trajectory = (
            "1 + 1 = 2 (left: 1 8 2)\n"
            "1 + 2 = 3 (left: 8 3)\n"
            "8 * 3 = 24 (left: 24)\n"
        )
        self.assertTrue(
            task.is_success_state("1 1 1 8", trajectory),
            "DEFECT: 8*(1+1+1)=24 is the canonical solution to '1 1 1 8' and "
            "must be scored successful regardless of `left:` print order",
        )

    def test_wrong_multiset_still_rejected(self):
        """Order-independence must NOT accept a genuinely wrong remain-set."""
        from federated_mcts.utils.format_check import check_final_result
        correct, _ = check_final_result(
            "1 + 2 = 3 (left: 4 5 5)", x="1 2 3 4",
        )
        self.assertFalse(
            correct,
            "DEFECT: remain-set {4,5,5} does not match {3,3,4} and must be rejected",
        )

    def test_sorted_and_unsorted_agree(self):
        """Permuting only the print order must not change the verdict."""
        from federated_mcts.utils.format_check import check_final_result
        verdicts = [
            check_final_result(f"1 + 2 = 3 (left: {order})", x="1 2 3 4")[0]
            for order in ("3 3 4", "4 3 3", "3 4 3")
        ]
        self.assertEqual(
            verdicts, [True, True, True],
            "DEFECT: `left:` print order must not affect the verdict",
        )


if __name__ == "__main__":
    unittest.main()
