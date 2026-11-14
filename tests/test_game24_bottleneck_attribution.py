"""Tests for post-hoc oracle bottleneck attribution.

The module recomputes reachability from recorded ``canonical_text`` so that
depth-level attribution is possible despite the oracle annotations being
discarded during a run.  These tests pin the classification semantics, since a
misclassification here would send Todo 9 at the wrong mechanism.
"""

import os
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_bottleneck_attribution import (
    LOSS_GENERATION_RECALL,
    LOSS_NO_CANDIDATE_GENERATED,
    LOSS_NONE,
    LOSS_SELECTION_REGRET,
    LOSS_TERMINAL_UNREACHED,
    attribute_run,
    attribute_task,
    candidate_reachability,
    load_task_metrics_from_journal,
    reaches_terminal,
    render_markdown,
)


def _cand(text, selected=False, stage="evaluator_scored"):
    return {
        "canonical_text": text,
        "is_selected": selected,
        "stage": stage,
        "raw_content": text,
    }


# A move that leaves {1,2,8}: reachable (1*2*8 = 16? no -- but see below).
# Ground truth used below was machine-checked against _can_reach_target_numbers.
REACHABLE = "1 + 1 = 2 (left: 1 2 8)\n"      # {1,2,8} -> reachable
UNREACHABLE = "1 - 1 = 0 (left: 0 1 8)\n"    # {0,1,8} -> unreachable


def _step(depth, events):
    return {"depth": depth, "candidates": events, "parent_count": 1}


def _task(task_id, solved, steps, task_input="1 1 1 8"):
    return {
        "task_id": task_id,
        "task_input": task_input,
        "solved": solved,
        "step_details": steps,
    }


class TestCandidateReachability(unittest.TestCase):
    def test_reachable_move_detected(self):
        self.assertIs(candidate_reachability(REACHABLE), True)

    def test_unreachable_move_detected(self):
        self.assertIs(candidate_reachability(UNREACHABLE), False)

    def test_empty_text_is_undefined_not_false(self):
        """An unparsable candidate must not be silently counted as unreachable."""
        self.assertIsNone(candidate_reachability(""))
        self.assertIsNone(candidate_reachability(None))
        self.assertIsNone(candidate_reachability("   "))

    def test_garbage_text_is_undefined(self):
        self.assertIsNone(candidate_reachability("no numbers here"))


class TestFirstLossClassification(unittest.TestCase):
    def test_solved_task_is_classified_none(self):
        t = _task("3", True, [
            _step(0, [_cand(REACHABLE, selected=True), _cand(UNREACHABLE)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_NONE)
        self.assertIsNone(a.first_loss_depth)

    def test_generation_recall_when_no_candidate_reachable(self):
        """Candidates exist but none is reachable -> generation recall."""
        t = _task("3", False, [
            _step(0, [_cand(UNREACHABLE, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_GENERATION_RECALL)
        self.assertEqual(a.first_loss_depth, 0)

    def test_selection_regret_when_reachable_existed_unselected(self):
        """A reachable candidate was present and the selector took another."""
        t = _task("3", False, [
            _step(0, [
                _cand(REACHABLE, selected=False),
                _cand(UNREACHABLE, selected=True),
            ]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_SELECTION_REGRET)
        self.assertEqual(a.first_loss_depth, 0)

    def test_no_candidate_generated_when_step_empty(self):
        t = _task("3", False, [_step(0, [])])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_NO_CANDIDATE_GENERATED)

    def test_first_loss_is_the_shallowest(self):
        """Depth 0 succeeded; depth 1 fails with no reachable candidate."""
        t = _task("3", False, [
            _step(0, [_cand(REACHABLE, selected=True)]),
            _step(1, [_cand(UNREACHABLE, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_depth, 1)
        self.assertEqual(a.first_loss_label, LOSS_GENERATION_RECALL)

    def test_reachable_selected_does_not_count_as_loss(self):
        """Selected set is non-empty and reachable -> no loss at that depth."""
        t = _task("3", False, [
            _step(0, [_cand(REACHABLE, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertIsNone(a.first_loss_depth)


class TestSelectionDetection(unittest.TestCase):
    def test_is_selected_flag_is_honoured(self):
        """The recorder sets is_selected but never a selected stage value."""
        t = _task("3", False, [
            _step(0, [_cand(REACHABLE, selected=True, stage="evaluator_scored")]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.depth_metrics[0].selected_count, 1)
        self.assertEqual(a.depth_metrics[0].selected_reachable_count, 1)

    def test_selected_stage_is_honoured_too(self):
        t = _task("3", False, [
            _step(0, [_cand(REACHABLE, selected=False, stage="selected")]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.depth_metrics[0].selected_count, 1)

    def test_unreachable_with_reachable_alternative_counted(self):
        t = _task("3", False, [
            _step(0, [
                _cand(REACHABLE, selected=False),
                _cand(UNREACHABLE, selected=True),
            ]),
        ])
        a = attribute_task(t)
        self.assertEqual(
            a.depth_metrics[0].unreachable_with_reachable_alternative_count, 1
        )

    def test_unreachable_without_alternative_not_counted(self):
        t = _task("3", False, [_step(0, [_cand(UNREACHABLE, selected=True)])])
        a = attribute_task(t)
        self.assertEqual(
            a.depth_metrics[0].unreachable_with_reachable_alternative_count, 0
        )


class TestRunAggregation(unittest.TestCase):
    def _run(self):
        return [
            _task("3", True, [_step(0, [_cand(REACHABLE, selected=True)])]),
            _task("9", False, [_step(0, [_cand(UNREACHABLE, selected=True)])]),
            _task("16", False, [
                _step(0, [
                    _cand(REACHABLE, selected=False),
                    _cand(UNREACHABLE, selected=True),
                ]),
            ]),
        ]

    def test_counts_partition_all_tasks(self):
        r = attribute_run(self._run())
        self.assertEqual(r.task_count, 3)
        self.assertEqual(r.solved_count, 1)
        self.assertEqual(r.unsolved_count, 2)
        total = sum(r.first_loss_counts.values())
        self.assertEqual(total, 3, "every task must land in exactly one class")

    def test_percentages_are_over_unsolved(self):
        r = attribute_run(self._run())
        pct = r.first_loss_percentages()
        self.assertAlmostEqual(pct[LOSS_GENERATION_RECALL], 50.0)
        self.assertAlmostEqual(pct[LOSS_SELECTION_REGRET], 50.0)
        self.assertNotIn(LOSS_NONE, pct)

    def test_depth_metrics_are_merged_across_tasks(self):
        r = attribute_run(self._run())
        self.assertEqual(len(r.depth_metrics), 1)
        dm = r.depth_metrics[0]
        self.assertEqual(dm.proposed_count, 4)
        self.assertEqual(dm.reachable_count, 2)
        self.assertEqual(dm.selected_count, 3)
        # task 3 selects a reachable candidate; task 9 selects an unreachable
        # one with no alternative; task 16 selects the unreachable one while a
        # reachable candidate sits unselected.  So exactly 1 selected-reachable.
        self.assertEqual(dm.selected_reachable_count, 1)
        self.assertEqual(dm.selected_unreachable_count, 2)

    def test_empty_run_is_safe(self):
        r = attribute_run([])
        self.assertEqual(r.task_count, 0)
        self.assertEqual(r.unsolved_count, 0)
        self.assertEqual(r.first_loss_percentages(), {
            LOSS_GENERATION_RECALL: 0.0,
            LOSS_SELECTION_REGRET: 0.0,
            LOSS_TERMINAL_UNREACHED: 0.0,
            LOSS_NO_CANDIDATE_GENERATED: 0.0,
        })


class TestTerminalUnreached(unittest.TestCase):
    """Every step picked a reachable candidate, yet {24} was never reached.

    This is the real pattern behind the 9 tasks that were otherwise
    unclassifiable: each individual decision looked solvable, and the chain
    still did not terminate on target.
    """

    TO_21 = ("7 / 7 = 1 (left: 1 3 7)\n3 * 7 = 21 (left: 1 21)\n"
             "1 * 21 = 21 (left: 21)\n")
    TO_36 = ("11 - 5 = 6 (left: 3 6 6)\n6 + 6 = 12 (left: 3 12)\n"
             "3 * 12 = 36 (left: 36)\n")
    TO_24 = ("1 + 1 = 2 (left: 1 2 8)\n2 * 8 = 16 (left: 1 16)\n"
             "8 + 16 = 24 (left: 24)\n")

    def test_reaches_terminal_detects_singleton_24(self):
        self.assertTrue(reaches_terminal(self.TO_24))

    def test_reaches_terminal_rejects_other_singletons(self):
        self.assertFalse(reaches_terminal(self.TO_21))
        self.assertFalse(reaches_terminal(self.TO_36))

    def test_reaches_terminal_rejects_multi_number_state(self):
        self.assertFalse(reaches_terminal(REACHABLE))

    def test_reachable_is_not_the_same_as_terminal(self):
        """The distinction that makes this class necessary."""
        self.assertTrue(candidate_reachability(REACHABLE))
        self.assertFalse(reaches_terminal(REACHABLE))

    def test_classified_terminal_unreached_when_every_step_had_reachable(self):
        """Every step has a reachable selection; none reaches {24}.

        ``REACHABLE`` = ``{1,2,8}`` is reachable but not terminal: it still holds
        three numbers, so it can be extended and is never a singleton 24.  With a
        reachable selection present at every depth, no per-step loss label
        applies, yet the chain never terminated on target -- exactly the pattern
        the nine real unclassifiable tasks show.
        """
        t = _task("1071", False, [
            _step(0, [_cand(REACHABLE, selected=True)]),
            _step(1, [_cand(REACHABLE, selected=True)]),
            _step(2, [_cand(REACHABLE, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_TERMINAL_UNREACHED)
        self.assertIsNone(a.first_loss_depth)
        for dm in a.depth_metrics:
            self.assertGreaterEqual(
                dm.selected_reachable_count, 1,
                "this scenario requires a reachable selection at every step",
            )

    def test_terminal_unreached_yields_to_a_reached_target(self):
        """If any selected chain reaches {24}, the task is not a loss."""
        t = _task("3", False, [
            _step(0, [_cand(REACHABLE, selected=True)]),
            _step(1, [_cand(self.TO_24, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_NONE)

    def test_solved_by_terminal_even_if_unsolved_flag_disagrees(self):
        """Reaching {24} is authoritative over a stale ``solved`` flag."""
        t = _task("3", False, [
            _step(0, [_cand(self.TO_24, selected=True)]),
        ])
        a = attribute_task(t)
        self.assertEqual(a.first_loss_label, LOSS_NONE)
        self.assertIsNone(a.first_loss_depth)

    def test_unselected_terminal_does_not_solve_the_task(self):
        """A terminal candidate that was NOT selected does not solve the task.

        With nothing selected at all, the first step shows a reachable candidate
        that the selector did not take, so ``selection_regret`` is the correct
        label -- the point being that the presence of a {24} candidate is not
        by itself enough.
        """
        t = _task("3", False, [
            _step(0, [_cand(self.TO_24, selected=False)]),
        ])
        a = attribute_task(t)
        self.assertNotEqual(a.first_loss_label, LOSS_NONE)
        self.assertEqual(a.first_loss_label, LOSS_SELECTION_REGRET)

    def test_every_unsolved_task_gets_a_loss_class(self):
        """No unsolved task may fall through to ``none``."""
        run = [
            _task("a", True, [_step(0, [_cand(self.TO_24, selected=True)])]),
            _task("b", False, [_step(0, [_cand(UNREACHABLE, selected=True)])]),
            _task("c", False, [
                _step(0, [_cand(REACHABLE, selected=False),
                          _cand(UNREACHABLE, selected=True)]),
            ]),
            _task("d", False, [
                _step(0, [_cand(REACHABLE, selected=True)]),
                _step(1, [_cand(self.TO_21, selected=True)]),
            ]),
        ]
        r = attribute_run(run)
        self.assertEqual(r.unsolved_count, 3)
        for t in r.tasks:
            if not t.solved:
                self.assertNotEqual(
                    t.first_loss_label, LOSS_NONE,
                    f"unsolved task {t.task_id} was classified {LOSS_NONE!r}",
                )
        self.assertEqual(r.first_loss_counts.get(LOSS_NONE), 1)


class TestJournalLoading(unittest.TestCase):
    def test_missing_metrics_raises_rather_than_shrinking_denominator(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "j.json")
            with open(p, "w") as f:
                json.dump({"_meta": {"run_fingerprint": "x"}, "3": {"ys": []}}, f)
            with self.assertRaises(ValueError) as ctx:
                load_task_metrics_from_journal(p)
            self.assertIn("task_metrics_json", str(ctx.exception))

    def test_meta_is_skipped(self):
        import json
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            p = os.path.join(tmp, "j.json")
            tm = _task("3", True, [_step(0, [_cand(REACHABLE, selected=True)])])
            with open(p, "w") as f:
                json.dump(
                    {"_meta": {"run_fingerprint": "x"},
                     "3": {"task_metrics_json": json.dumps(tm)}},
                    f,
                )
            loaded = load_task_metrics_from_journal(p)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0]["task_id"], "3")


class TestMarkdown(unittest.TestCase):
    def test_render_includes_classes_and_depths(self):
        r = attribute_run([
            _task("3", False, [_step(0, [_cand(UNREACHABLE, selected=True)])]),
        ])
        md = render_markdown(r)
        self.assertIn("First-loss classification", md)
        self.assertIn(LOSS_GENERATION_RECALL, md)
        self.assertIn("| depth |", md)


if __name__ == "__main__":
    unittest.main()
