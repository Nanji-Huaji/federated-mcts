"""Tests for the tree replay and the pinned, real 148-puzzle archive (no model)."""

import hashlib
import json
import unittest
from decimal import Decimal
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]

if __package__:
    from . import tree_archive as tree
else:
    import tree_archive as tree


def trajectory(*steps):
    # Hand-worked parser cases, not experimental data or simulated model replies.
    return "\n".join(f"{a} {op} {b} = {v} (left: {' '.join(map(str, left))})"
                     for a, op, b, v, left in steps)


class ReplayTests(unittest.TestCase):
    def test_consumes_duplicate_leaves_and_accepts_exact_zero(self):
        nums = [14, 14, 12, 2]
        y = trajectory((14, "-", 14, 0, [12, 0, 2]),
                       (12, "+", 0, 12, [2, 12]), (12, "*", 2, 24, [24]))
        r = tree.replay_trajectory(nums, y)
        self.assertTrue(r["valid"])
        self.assertEqual(r["reason"], "valid")
        self.assertRegex(r["solution_sha256"], r"^[a-f0-9]{64}$")
        # A single terminal newline is the archive's encoding, not a fourth step.
        lf = tree.replay_trajectory(nums, y + "\n")
        self.assertEqual(r["solution_sha256"], lf["solution_sha256"])
        self.assertNotEqual(r["raw_sha256"], lf["raw_sha256"])


    def test_rejects_invalid_arithmetic_or_multisets_without_salvage(self):
        nums = [14, 14, 12, 2]
        s1 = (14, "-", 14, 0, [12, 0, 2])
        s2 = (12, "+", 0, 12, [2, 12])
        s3 = (12, "*", 2, 24, [24])
        for why, steps in [
            ("unavailable_operand", ((12, "-", 12, 0, [14, 2, 0]), s2, s3)),
            ("wrong_result", ((14, "-", 14, 1, [12, 1, 2]), s2, s3)),
            ("wrong_left_multiset", ((14, "-", 14, 0, [2, 0, 2]), s2, s3)),
            ("wrong_left_multiset", (s1, s2, (12, "*", 2, 24, [24, 24]))),
            ("negative_result", ((2, "-", 12, 0, [14, 14, 0]), s2, s3)),
            ("division_by_zero", (s1, (12, "/", 0, 12, [2, 12]), s3)),
            ("inexact_division", ((14, "/", 12, 1, [14, 2, 1]), s2, s3)),
            ("not_24", (s1, s2, (12, "+", 2, 14, [14]))),
        ]:
            with self.subTest(reason=why):
                r = tree.replay_trajectory(nums, trajectory(*steps))
                self.assertEqual((r["valid"], r["reason"], r["solution_sha256"]), (False, why, None))

    def test_rejects_non_ascii_non_integral_or_extra_text_and_never_returns_it(self):
        nums = [14, 14, 12, 2]
        y = trajectory((14, "-", 14, 0, [12, 0, 2]),
                       (12, "+", 0, 12, [2, 12]), (12, "*", 2, 24, [24]))
        for text, reason in [(y.replace("14", "١٤"), "format"), (y.replace("14", "14.0"), "format"),
                             (y.replace("14", "14/1"), "format"), (y.replace("14", "-14"), "format"),
                             (y.replace("\n", "\r\n"), "format"), (y + "\n\n", "step_count"),
                             (y.replace("\n", r"\n"), "step_count"), ("comment: " + y, "format"),
                             (y.rsplit("\n", 1)[0], "step_count"), (y + "\n24", "step_count"),
                             (chr(0xD800) + y, "format")]:
            with self.subTest(reason=reason):
                r = tree.replay_trajectory(nums, text)
                self.assertFalse(r["valid"])
                self.assertEqual(r["reason"], reason)
                self.assertRegex(r["raw_sha256"], r"^[0-9a-f]{64}$")
                self.assertFalse(any(v == text for v in r.values()), "raw payload escaped")
        for bad_inputs in ([-1, 14, 12, 2], [True, 14, 12, 2], [14, 12, 2]):
            with self.assertRaises(ValueError):
                tree.replay_trajectory(bad_inputs, y)

    def test_exact_solutions_distinguish_order_from_expression_provenance(self):
        nums = [27, 9, 7, 1]
        a = trajectory((27, "/", 9, 3, [3, 1, 7]), (7, "+", 1, 8, [3, 8]), (3, "*", 8, 24, [24]))
        b = trajectory((1, "+", 7, 8, [8, 9, 27]), (27, "/", 9, 3, [3, 8]), (8, "*", 3, 24, [24]))
        left, right = (tree.replay_trajectory(nums, y) for y in (a, b))
        self.assertTrue(left["valid"] and right["valid"])
        self.assertNotEqual(left["solution_sha256"], right["solution_sha256"])
        self.assertEqual(left["expression_set_sha256"], right["expression_set_sha256"])
        self.assertEqual(left["n_possible_expressions"], 1)
        # At step two, a newly-made 6 is indistinguishable numerically from an input 6.
        ambiguous = trajectory((10, "-", 4, 6, [2, 6, 6]), (6, "-", 2, 4, [4, 6]), (4, "*", 6, 24, [24]))
        self.assertEqual(tree.replay_trajectory([10, 4, 6, 2], ambiguous)["n_possible_expressions"], 2)


class ArchiveTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.archive = tree.load_tree_archive(ROOT)

    def test_real_input_order_retained_slots_and_strict_outcomes(self):
        runs = self.archive["runs"]
        self.assertEqual([r["idx"] for r in runs], list(range(148)))
        self.assertEqual([r["dataset_rank"] for r in runs],
                         [*range(100, 200), *range(500, 548)])
        input_order = [sorted(r["input"]) for r in runs]
        self.assertEqual(hashlib.sha256(json.dumps(input_order, separators=(",", ":")).encode()).hexdigest(),
                         "db473ae0eee33f635c3434b507e70cf882e99108f5c34464bcfbf04f0f04433d")
        self.assertEqual({r["idx"]: len(r["candidates"]) for r in runs if len(r["candidates"]) < 10},
                         {14: 9, 75: 3, 89: 3, 142: 3})
        summary = self.archive["summary"]
        self.assertEqual(summary["solved_at"], {"1": 87, "3": 124, "full": 132})
        self.assertEqual(summary["selected_slot_verdicts"], {"step_count": 103, "not_24": 746, "valid": 609})


    def test_real_trace_run_slot_and_expression_duplication(self):
        s = self.archive["summary"]
        self.assertEqual(s["n_archive_files"], 1)
        d = s["duplication"]
        for key, (total, distinct) in {
            "input_indices": (148, 148), "input_multisets": (148, 148), "records_verbatim": (148, 148),
            "ordered_search_traces": (148, 148), "ordered_selected_vectors": (148, 148),
            "proposal_pools": (8298, 8181), "final_pool": (2286, 2285),
            "all_selected_frontiers": (3897, 3797), "retained_trajectories": (1458, 1458),
            "valid_ordered_solutions": (609, 609), "valid_expression_sets": (609, 575),
        }.items():
            with self.subTest(key=key):
                self.assertEqual((d[key]["occurrences"], d[key]["distinct"]), (total, distinct))
                self.assertEqual(d[key]["duplicate_excess"], total - distinct)
        self.assertEqual(d["valid_expression_sets"]["multiplicity_histogram"], {"1": 541, "2": 34})
        self.assertEqual(d["ambiguous_expression_slots"], 35)
        coll, = d["within_pool_collisions"]
        self.assertEqual((coll["idx"], coll["depth"], coll["pool_slots"]), (55, 2, [2, 3]))
        self.assertEqual(s["search_by_depth"][0]["parent_slots"], 148)
        self.assertEqual([v["proposal_slots"] for v in s["search_by_depth"]], [981, 5031, 2286])
        self.assertEqual([v["carry_slots"] for v in s["search_by_depth"]], [0, 6, 110])
        self.assertEqual(sum(v["unlinked_slots"] for v in s["search_by_depth"]), 0)

    def test_usage_is_adjacent_deltas_not_sum_of_cumulative_snapshots(self):
        runs = self.archive["runs"]
        fields = ["total_calls", "prompt_tokens", "completion_tokens", "total_tokens", "cost"]
        self.assertEqual([runs[0]["budget"][k] for k in fields], [113, 66193, 4609, 70802, "0.075411"])
        self.assertEqual([runs[-1]["budget"][k] for k in fields], [127, 82774, 6086, 88860, "0.094946"])
        summary = self.archive["summary"]["budget"]
        self.assertEqual([summary[k] for k in fields], [16048, 9433188, 749643, 10182831, "10.932474"])
        self.assertEqual(sum(Decimal(r["budget"]["cost"]) for r in runs), Decimal("10.932474"))
        self.assertEqual(summary["cost_unit"], "CNY")
        self.assertIsNone(summary["currency_field_in_json"])
        self.assertIsNone(summary["edge_usage"])
        self.assertIsNone(summary["prefix_usage"])
        self.assertIsNone(summary["wall_seconds"])


if __name__ == "__main__":
    unittest.main()
