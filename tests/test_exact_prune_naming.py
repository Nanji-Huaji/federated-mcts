"""Test that --game24_exact_prune adds _exactprune_True suffix to output filenames
in both root merged_run.py and scripts/merged_run.py, and that the default
filename is unchanged.
"""

import ast
import json
import os
import sys
import textwrap
import unittest
from unittest.mock import patch


def _extract_file_name_generater_body(source_path):
    """Parse a merged_run.py and return the body source text of
    file_name_generater (excluding the def line and outer indent)."""
    with open(source_path) as f:
        source = f.read()
    tree = ast.parse(source)
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "file_name_generater":
            body_lines = source.splitlines()
            # node.body[0].lineno is the first line of the function body
            # node.end_lineno is the last line of the function
            first_body_line = node.body[0].lineno  # 1-indexed
            last_line = node.end_lineno  # 1-indexed
            # Extract just the body (skip the def line)
            body = "\n".join(body_lines[first_body_line - 1 : last_line])
            return textwrap.dedent(body)
    raise RuntimeError(f"file_name_generater not found in {source_path}")


def _make_fake_args(**overrides):
    """Build a minimal fake args namespace."""

    class FakeArgs:
        pass

    args = FakeArgs()
    args.task = "game24"
    args.solve_method = "tot"
    args.localbackend = "local-model"
    args.remotebackend = "remote-model"
    args.temperature = 0.7
    args.method_generate = "sample"
    args.n_generate_sample = 5
    args.method_evaluate = "vote"
    args.n_evaluate_sample = 3
    args.method_select = "greedy"
    args.n_select_sample = 1
    args.task_start_index = 0
    args.task_end_index = 10
    args.slm_generate = False
    args.slm_eval = False
    args.check_format = False
    args.eval_rule = "standard"
    args.warm_start = False
    args.last_lm = "none"
    args.inference_idx = 0
    args.naive_run = False
    args.prompt_sample = "standard"
    args.search_policy = "baseline"
    # model_config omitted so hasattr returns False (skip branch)
    for k, v in overrides.items():
        setattr(args, k, v)
    return args


def _call_file_name_generater(source_path, args):
    """Extract file_name_generater from source_path, wrap its body in a
    function, exec, and call with args.  Returns the generated filename."""
    body = _extract_file_name_generater_body(source_path)
    wrapper = "def file_name_generater(args):\n" + textwrap.indent(body, "    ")
    ns = {"json": json, "os": os, "getattr": getattr}
    exec(wrapper, ns)
    return ns["file_name_generater"](args)


class TestExactPruneNamingRoot(unittest.TestCase):
    """Test file_name_generater in root merged_run.py."""

    ROOT_PATH = "/home/tiantianyi/code/federated-mcts/merged_run.py"

    def test_default_no_suffix(self):
        args = _make_fake_args(game24_exact_prune=False)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.ROOT_PATH, args)
        self.assertNotIn("_exactprune_True", fname,
                         "DEFECT: exact-prune suffix present when flag is False")
        self.assertIn("game24", fname)

    def test_exact_prune_true_adds_suffix(self):
        args = _make_fake_args(game24_exact_prune=True)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.ROOT_PATH, args)
        self.assertTrue(fname.endswith("_exactprune_True"),
                        f"DEFECT: filename should end with _exactprune_True, got: {fname}")
        self.assertEqual(fname.count("_exactprune_True"), 1)

    def test_no_game24_exact_prune_attr_no_suffix(self):
        args = _make_fake_args()
        if hasattr(args, "game24_exact_prune"):
            delattr(args, "game24_exact_prune")
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.ROOT_PATH, args)
        self.assertNotIn("_exactprune_True", fname,
                         "DEFECT: suffix present when game24_exact_prune attr is absent")

    def test_suffix_position(self):
        args = _make_fake_args(game24_exact_prune=True)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.ROOT_PATH, args)
        self.assertTrue(fname.endswith("_exactprune_True"),

                        f"DEFECT: filename should end with _exactprune_True, got: {fname}")

class TestExactPruneNamingScripts(unittest.TestCase):
    """Test file_name_generater in scripts/merged_run.py."""

    SCRIPT_PATH = "/home/tiantianyi/code/federated-mcts/scripts/merged_run.py"

    def test_default_no_suffix(self):
        args = _make_fake_args(game24_exact_prune=False)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.SCRIPT_PATH, args)
        self.assertNotIn("_exactprune_True", fname,
                         "DEFECT: exact-prune suffix present when flag is False")

    def test_exact_prune_true_adds_suffix(self):
        args = _make_fake_args(game24_exact_prune=True)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.SCRIPT_PATH, args)
        self.assertTrue(fname.endswith("_exactprune_True"),
                        f"DEFECT: filename should end with _exactprune_True, got: {fname}")

    def test_no_game24_exact_prune_attr_no_suffix(self):
        args = _make_fake_args()
        if hasattr(args, "game24_exact_prune"):
            delattr(args, "game24_exact_prune")
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.SCRIPT_PATH, args)
        self.assertNotIn("_exactprune_True", fname,
                         "DEFECT: suffix present when game24_exact_prune attr is absent")

    def test_naive_run_with_exact_prune(self):
        args = _make_fake_args(game24_exact_prune=True, naive_run=True)
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.SCRIPT_PATH, args)
        self.assertTrue(fname.endswith("_exactprune_True"),
                        f"DEFECT: naive_run path missing suffix, got: {fname}")

    def test_policy_suffix_and_exact_prune_coexist(self):
        args = _make_fake_args(game24_exact_prune=True, search_policy="diverse")
        with patch("os.makedirs"):
            fname = _call_file_name_generater(self.SCRIPT_PATH, args)
        self.assertIn("_policy_diverse", fname,
                      "DEFECT: policy suffix missing")
        self.assertTrue(fname.endswith("_exactprune_True"),
                        "DEFECT: exact-prune suffix should be last")
        policy_idx = fname.index("_policy_diverse")
        prune_idx = fname.index("_exactprune_True")
        self.assertLess(policy_idx, prune_idx,
                        "DEFECT: policy before exact-prune")


if __name__ == "__main__":
    unittest.main()
