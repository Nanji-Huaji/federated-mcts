import io
import sys
import unittest
from contextlib import redirect_stdout

sys.path.insert(0, "src")

from federated_mcts.tasks.game24 import Game24Task


class TestGame24VotePrompt(unittest.TestCase):
    """Regression tests for vote_prompt_wrap and vote_outputs_unwrap."""

    def test_vote_prompt_renders_concrete_candidates_not_literal_placeholders(self):
        """vote_prompt_wrap must interpolate {i} and {y}, not emit literal braces."""
        ys = ["1 + 2 = 3 (left: 3 3 4)", "3 * 4 = 12 (left: 1 12)"]
        prompt = Game24Task.vote_prompt_wrap("1 2 3 4", ys)

        self.assertNotIn("{i}", prompt,
                         "DEFECT: vote_prompt_wrap emits literal {i}")
        self.assertNotIn("{y}", prompt,
                         "DEFECT: vote_prompt_wrap emits literal {y}")

        self.assertIn("Choice 0:", prompt,
                      "DEFECT: missing Choice 0: label")
        self.assertIn("Choice 1:", prompt,
                      "DEFECT: missing Choice 1: label")
        self.assertIn("1 + 2 = 3 (left: 3 3 4)", prompt,
                      "DEFECT: missing first candidate content")
        self.assertIn("3 * 4 = 12 (left: 1 12)", prompt,
                      "DEFECT: missing second candidate content")

    def test_vote_prompt_candidate_ids_are_zero_based(self):
        ys = ["a", "b", "c"]
        prompt = Game24Task.vote_prompt_wrap("1 2 3 4", ys)
        self.assertIn("Choice 0:", prompt)
        self.assertIn("Choice 1:", prompt)
        self.assertIn("Choice 2:", prompt)
        self.assertNotIn("Choice 3:", prompt)

    def test_vote_prompt_template_shows_literal_s_placeholder(self):
        ys = ["a"]
        prompt = Game24Task.vote_prompt_wrap("1 2 3 4", ys)
        self.assertIn("{s}", prompt,
                      "The prompt should contain literal '{s}' as instruction")

    def test_vote_prompt_empty_candidates(self):
        prompt = Game24Task.vote_prompt_wrap("1 2 3 4", [])
        self.assertIn("Choices:", prompt)
        parts = prompt.split("Choices:")
        self.assertEqual(len(parts), 2)
        self.assertEqual(parts[1].strip(), "",
                         "Empty candidates should produce empty Choices section")

    def test_vote_prompt_forbids_analysis_before_conclusion(self):
        """vote_prompt must NOT ask for analysis before parseable line,
        so short max_tokens responses never truncate the best choice."""
        from federated_mcts.prompts.game24 import vote_prompt
        prompt_lower = vote_prompt.lower()
        self.assertNotIn("analyze", prompt_lower,
            "DEFECT: vote_prompt asks for analysis which may be truncated before conclusion")
        self.assertNotIn("conclude in the last line", prompt_lower,
            "DEFECT: vote_prompt tells model to put conclusion last (should be first or only)")
        self.assertIn("the best choice is", prompt_lower,
            "vote_prompt must include the parseable marker")



class TestGame24VoteOutputsUnwrap(unittest.TestCase):
    """Regression tests for vote_outputs_unwrap parsing and diagnostics."""

    def test_parses_best_choice_is_N(self):
        vote_outputs = ["The best choice is 0", "The best choice is 2"]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 4)
        self.assertEqual(results, [1, 0, 1, 0])

    def test_parses_case_insensitive(self):
        vote_outputs = ["the BEST choice is 1"]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 3)
        self.assertEqual(results, [0, 1, 0])

    def test_parses_multiline_with_best_choice_at_end(self):
        vote_outputs = [
            "Choice 0 does X. Choice 1 does Y.\nThe best choice is 1"
        ]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 3)
        self.assertEqual(results, [0, 1, 0])

    def test_out_of_range_vote_ignored(self):
        vote_outputs = ["The best choice is 5"]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 3)
        self.assertEqual(results, [0, 0, 0])

    def test_negative_vote_ignored(self):
        vote_outputs = ["The best choice is -1"]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 3)
        self.assertEqual(results, [0, 0, 0])

    def test_no_match_prints_actual_output_not_literal_braces(self):
        vote_outputs = ["garbage response with no match"] * 3
        buf = io.StringIO()
        with redirect_stdout(buf):
            results = Game24Task.vote_outputs_unwrap(vote_outputs, 2)
        output = buf.getvalue()
        self.assertNotIn("{[vote_output]}", output,
                         "DEFECT: diagnostic prints literal braces")
        self.assertIn("garbage response", output,
                      "DEFECT: diagnostic should print actual output")
        self.assertEqual(output.count("vote no match:"), 3)
        self.assertEqual(results, [0, 0])

    def test_mixed_valid_and_invalid_votes(self):
        vote_outputs = [
            "The best choice is 0",
            "nonsense",
            "The best choice is 0",
        ]
        buf = io.StringIO()
        with redirect_stdout(buf):
            results = Game24Task.vote_outputs_unwrap(vote_outputs, 2)
        self.assertEqual(results, [2, 0])
        self.assertIn("nonsense", buf.getvalue())
    def test_vote_outputs_unwrap_parses_standalone_best_choice_line(self):
        vote_outputs = ["The best choice is 2"]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 4)
        self.assertEqual(results, [0, 0, 1, 0],
            "DEFECT: standalone parseable line not recognized")

    def test_vote_outputs_unwrap_parses_best_choice_with_analysis_after(self):
        vote_outputs = [
            "The best choice is 1\n\nChoice 0 does X. Choice 1 does Y."
        ]
        results = Game24Task.vote_outputs_unwrap(vote_outputs, 3)
        self.assertEqual(results, [0, 1, 0],
            "DEFECT: parseable line followed by analysis not recognized")




if __name__ == "__main__":
    unittest.main()
