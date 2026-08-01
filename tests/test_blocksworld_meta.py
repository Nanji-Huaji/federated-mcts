"""Meta tests for the PlanBench Blocksworld task: splits, registries, CLI
choices, and the checked-generation empty-valid-candidate fallback."""

import json
import os
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, "src")

from federated_mcts.tasks import get_task
from federated_mcts.tasks.blocksworld import BlocksworldTask

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_COUNT = 501
SPLIT_PATH = os.path.join(REPO_ROOT, "data", "splits", "blocksworld.json")


class ArgsStub:
    check_format = True


class TestSplits(unittest.TestCase):
    def test_split_file_covers_all_records_disjointly(self):
        self.assertTrue(os.path.exists(SPLIT_PATH), SPLIT_PATH)
        with open(SPLIT_PATH) as fh:
            split = json.load(fh)
        self.assertEqual(set(split), {"train", "val", "test"})
        for part in split.values():
            self.assertTrue(all(0 <= i < DATA_COUNT for i in part))
        self.assertEqual(
            set(split["train"]) | set(split["val"]) | set(split["test"]),
            set(range(DATA_COUNT)),
        )
        self.assertEqual(set(split["train"]) & set(split["val"]), set())
        self.assertEqual(set(split["train"]) & set(split["test"]), set())
        self.assertEqual(set(split["val"]) & set(split["test"]), set())

    def test_split_proportions_are_70_15_15(self):
        with open(SPLIT_PATH) as fh:
            split = json.load(fh)
        total = sum(len(v) for v in split.values())
        self.assertEqual(total, DATA_COUNT)
        self.assertAlmostEqual(len(split["train"]) / total, 0.70, delta=0.01)
        self.assertAlmostEqual(len(split["val"]) / total, 0.15, delta=0.01)
        self.assertAlmostEqual(len(split["test"]) / total, 0.15, delta=0.01)


class TestRegistry(unittest.TestCase):
    def test_federated_mcts_registry(self):
        self.assertIsInstance(get_task("blocksworld"), BlocksworldTask)

    def test_tot_registry_resolves_blocksworld(self):
        from tot.tasks import get_task as tot_get_task

        self.assertIsInstance(tot_get_task("blocksworld"), BlocksworldTask)

    def test_cli_choices_include_blocksworld(self):
        for runner in ("merged_run.py", os.path.join("scripts", "merged_run.py")):
            path = os.path.join(REPO_ROOT, runner)
            with open(path) as fh:
                text = fh.read()
            self.assertIn('"blocksworld"', text, runner)


class TestCheckedGenerationEmptyFallback(unittest.TestCase):
    def test_blocksworld_invalid_branch_becomes_unexpandable(self):
        from federated_mcts.core.generation import get_proposals_with_check

        task = BlocksworldTask()
        x = task.get_input(0)
        with patch(
            "federated_mcts.core.generation.gpt",
            return_value=["pick up z\n\ncommentary junk"],
        ):
            result = get_proposals_with_check(ArgsStub(), 0, task, x, "", client=None)
        self.assertEqual(result, [])

    def test_task_without_hook_keeps_old_behavior(self):
        from federated_mcts.core.generation import get_proposals_with_check

        class PlainTask:
            @staticmethod
            def pre_generate_check(y):
                return True

            @staticmethod
            def propose_prompt_wrap(x, y):
                return "prompt"

            @staticmethod
            def process_generate_result(pro, x, y, check):
                return False, y

        with patch(
            "federated_mcts.core.generation.gpt",
            return_value=["only junk here"],
        ):
            result = get_proposals_with_check(
                ArgsStub(), 0, PlainTask(), "x", "y", client=None
            )
        self.assertEqual(result, ["y"])


if __name__ == "__main__":
    unittest.main()
