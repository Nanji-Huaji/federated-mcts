"""Tests for the FOLIO WikiLogic / HybLogic tasks."""

import json
import os
import sys
import unittest

sys.path.insert(0, "src")

from federated_mcts.tasks import get_task
from federated_mcts.tasks.folio import (
    FOLIOBaseTask,
    HybLogicTask,
    WikiLogicTask,
    extract_label,
)

TRAIN_COUNT = 1004
VALIDATION_COUNT = 204
WIKI_TRAIN_COUNT = 537
HYB_TRAIN_COUNT = 467
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _split_path(name: str) -> str:
    return os.path.join(REPO_ROOT, "data", "splits", f"{name}.json")


class TestDataLoading(unittest.TestCase):
    def test_train_data_has_expected_fields(self):
        task = FOLIOBaseTask()
        self.assertEqual(len(task), TRAIN_COUNT)
        required = {"premises", "conclusion", "label", "source", "story_id", "example_id"}
        for item in task.data[:5]:
            self.assertTrue(required.issubset(item.keys()))
            self.assertIn(item["label"], ("True", "False", "Unknown"))
            self.assertIn(item["source"], ("WikiLogic", "HybLogic"))

    def test_validation_file_loads(self):
        task = FOLIOBaseTask(file="folio-validation.jsonl")
        self.assertEqual(len(task), VALIDATION_COUNT)
        self.assertTrue(all(item["label"] in ("True", "False", "Unknown") for item in task.data))

    def test_get_input_format(self):
        task = FOLIOBaseTask()
        x = task.get_input(0)
        self.assertTrue(x.startswith("Premises:\n- "))
        self.assertIn("\n\nConclusion: ", x)


class TestSourceFiltering(unittest.TestCase):
    def test_wikilogic_filters_by_source(self):
        task = WikiLogicTask()
        self.assertEqual(len(task), WIKI_TRAIN_COUNT)
        self.assertTrue(all(item["source"] == "WikiLogic" for item in task.data))

    def test_hyblogic_filters_by_source(self):
        task = HybLogicTask()
        self.assertEqual(len(task), HYB_TRAIN_COUNT)
        self.assertTrue(all(item["source"] == "HybLogic" for item in task.data))

    def test_base_keeps_both_sources(self):
        task = FOLIOBaseTask()
        self.assertEqual({item["source"] for item in task.data}, {"WikiLogic", "HybLogic"})

    def test_get_task_registration(self):
        self.assertIsInstance(get_task("wikilogic"), WikiLogicTask)
        self.assertIsInstance(get_task("hyblogic"), HybLogicTask)


class TestLabelParsing(unittest.TestCase):
    def test_extract_label_variants(self):
        cases = [
            ("...\n#### True", "True"),
            ("...\n#### FALSE.", "False"),
            ("...\n#### unknown", "Unknown"),
            ("...\n#### Uncertain", "Unknown"),
            ("no marker here", None),
        ]
        for output, expected in cases:
            self.assertEqual(extract_label(output), expected)

    def test_test_output_modify_compares_to_label(self):
        task = FOLIOBaseTask()
        idx = next(i for i, item in enumerate(task.data) if item["label"] == "True")
        r, out = task.test_output_modify(idx, "derived.\n#### True")
        self.assertEqual(r, {"r": 1})
        self.assertEqual(out, "derived.\n#### True")
        r_wrong, _ = task.test_output_modify(idx, "derived.\n#### False")
        self.assertEqual(r_wrong, {"r": 0})


class TestCanonicalState(unittest.TestCase):
    def test_state_key_is_stripped_line_tuple(self):
        task = FOLIOBaseTask()
        a = task.canonical_state_key("x", "  line1  \nline2\n\n")
        b = task.canonical_state_key("x", "line1\nline2")
        self.assertEqual(a, ("line1", "line2"))
        self.assertEqual(a, b)

    def test_different_line_sets_are_distinct(self):
        task = FOLIOBaseTask()
        seen = {task.canonical_state_key("x", "a\nb")}
        self.assertIn(task.canonical_state_key("x", "a\nb"), seen)
        self.assertNotIn(task.canonical_state_key("x", "a\nc"), seen)


class TestSuccessState(unittest.TestCase):
    def test_success_requires_marker_and_label_match(self):
        task = FOLIOBaseTask()
        idx = next(i for i, item in enumerate(task.data) if item["label"] == "Unknown")
        task.get_input(idx)
        self.assertTrue(task.is_success_state("x", "premise1\npremise2\n#### Unknown"))
        self.assertFalse(task.is_success_state("x", "premise1\npremise2\n#### True"))
        self.assertFalse(task.is_success_state("x", "premise1\npremise2"))


class TestPromptWraps(unittest.TestCase):
    def test_propose_prompt_contains_input_and_progress(self):
        task = FOLIOBaseTask()
        prompt = task.propose_prompt_wrap(task.get_input(0), "derived fact\n")
        self.assertIn("derived fact", prompt)
        self.assertIn("Conclusion", prompt)
        self.assertIn("Possible next steps", prompt)

    def test_propose_prompt_reuses_cot_when_done(self):
        task = FOLIOBaseTask()
        prompt = task.propose_prompt_wrap("x", "step one\n#### True")
        self.assertIn("Steps:", prompt)

    def test_value_prompt_contains_progress(self):
        task = FOLIOBaseTask()
        prompt = task.value_prompt_wrap("x", "step one")
        self.assertIn("step one", prompt)
        self.assertIn("Judge:", prompt)

    def test_value_outputs_unwrap_matches_gsm8k_convention(self):
        task = FOLIOBaseTask()
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["sure"]), 20)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["likely", "sure"]), 21)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["unlikely"]), 0.1)
        self.assertEqual(task.value_outputs_unwrap("x", "y", ["impossible"]), 0.001)

    def test_joint_rank_prompt_requests_json_ranking(self):
        prompt = FOLIOBaseTask.joint_rank_prompt_wrap("x", ["cand A\nstep1", "cand B\nstep2"])
        self.assertIn('"ranking"', prompt)
        self.assertIn("0: step1", prompt)
        self.assertIn("1: step2", prompt)

    def test_generate_hooks(self):
        task = FOLIOBaseTask()
        self.assertTrue(task.pre_generate_check("some steps"))
        self.assertFalse(task.pre_generate_check("some steps\n#### True"))
        ok, out = task.process_generate_result(" new fact ", "x", "old\n", True)
        self.assertTrue(ok)
        self.assertEqual(out, "old\nnew fact\n")
        ok_empty, _ = task.process_generate_result("   ", "x", "old\n", True)
        self.assertFalse(ok_empty)


class TestSplits(unittest.TestCase):
    def test_split_files_cover_each_subset(self):
        for name, count in (("wikilogic", WIKI_TRAIN_COUNT), ("hyblogic", HYB_TRAIN_COUNT)):
            path = _split_path(name)
            self.assertTrue(os.path.exists(path), path)
            with open(path) as fh:
                split = json.load(fh)
            self.assertEqual(set(split), {"train", "val", "test"})
            for part in split.values():
                self.assertTrue(all(0 <= i < count for i in part))
            self.assertEqual(sum(len(v) for v in split.values()), count)

    def test_split_proportions_are_70_15_15(self):
        for name in ("wikilogic", "hyblogic"):
            with open(_split_path(name)) as fh:
                split = json.load(fh)
            total = sum(len(v) for v in split.values())
            self.assertAlmostEqual(len(split["train"]) / total, 0.70, delta=0.01)
            self.assertAlmostEqual(len(split["val"]) / total, 0.15, delta=0.01)
            self.assertAlmostEqual(len(split["test"]) / total, 0.15, delta=0.01)


if __name__ == "__main__":
    unittest.main()
