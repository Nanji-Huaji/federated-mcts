"""Regression: a run without ``--task_split`` must complete, not crash.

The retired ``scripts/merged_run.py`` had a function-local ``import json`` that
shadowed the module-level import, so a run without ``--task_split`` raised
``UnboundLocalError`` at the first output write.  This test drives the
canonical CLI itself and asserts the no-split path exits cleanly without
writing anything.
"""

from __future__ import annotations

import contextlib
import io
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, "src")

from federated_mcts.cli import run_main

ROOT = Path(__file__).resolve().parents[1]


_MODEL_CONFIG = [
    {
        "client_name": "local_client",
        "model": "Qwen/Qwen2.5-32B-Instruct",
        "max_tokens": 256,
        "api_key": "test-key",
        "api_base": "http://127.0.0.1:9/v1",
    },
    {
        "client_name": "remote_client",
        "model": "deepseek-v4-flash",
        "max_tokens": 512,
        "api_key": "test-key",
        "api_base": "http://127.0.0.1:9/v1",
    },
]


class TestCliWithoutTaskSplit(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmpdir = self._tmp.name
        self._cwd = os.getcwd()
        os.chdir(self.tmpdir)
        os.makedirs("configs")
        with open("configs/model_config.json", "w", encoding="utf-8") as stream:
            json.dump(_MODEL_CONFIG, stream)

    def tearDown(self) -> None:
        os.chdir(self._cwd)
        self._tmp.cleanup()

    def test_dry_run_without_task_split_exits_zero(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            exit_code = run_main(
                [
                    "--task",
                    "blocksworld",
                    "--task_start_index",
                    "0",
                    "--task_end_index",
                    "2",
                    "--model_config",
                    "configs/model_config.json",
                    "--dry_run",
                ]
            )

        self.assertEqual(exit_code, 0, output.getvalue())
        self.assertIn("DRY RUN", output.getvalue())
        self.assertIn("task IDs (2): [0, 1]", output.getvalue())

    def test_dry_run_without_task_split_writes_nothing(self) -> None:
        before = sorted(os.listdir("."))
        with contextlib.redirect_stdout(io.StringIO()):
            run_main(
                [
                    "--task",
                    "blocksworld",
                    "--task_start_index",
                    "0",
                    "--task_end_index",
                    "2",
                    "--model_config",
                    "configs/model_config.json",
                    "--dry_run",
                ]
            )
        self.assertEqual(sorted(os.listdir(".")), before)
        self.assertFalse(os.path.exists("logs"))

    def test_module_entry_point_without_task_split(self) -> None:
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src")
        env["RUN_LIVE_DEEPSEEK_DIAGNOSTIC"] = "0"
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "federated_mcts",
                "--task",
                "blocksworld",
                "--task_start_index",
                "0",
                "--task_end_index",
                "2",
                "--model_config",
                "configs/model_config.json",
                "--dry_run",
            ],
            cwd=self.tmpdir,
            env=env,
            capture_output=True,
            text=True,
            timeout=180,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("DRY RUN", result.stdout)
        self.assertIn("task IDs (2): [0, 1]", result.stdout)
        self.assertFalse(os.path.exists(os.path.join(self.tmpdir, "logs")))


if __name__ == "__main__":
    unittest.main()
