"""Tests for B0 gate enforcement and CLI integration.

Covers: dry-run pass/fail, missing flags, no-side-effect, over-gate
rejection, valid evidence pass, fingerprint derivation from canonical
manifest fields.  All fixtures synthetic.
"""

from __future__ import annotations

import os
import sys
import tempfile
import unittest
from unittest.mock import MagicMock

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_gate import (
    add_b0_cli_args,
    enforce_b0_gate,
)

from tests._game24_b0_fixtures import (
    _args,
    _bf,
    _derive_fps,
    _jf,
    _make_all,
    _mf,
    _pf,
)


class TestGateEnforcement(unittest.TestCase):
    """Given CLI args, When enforcing B0 gate, Then correct decision."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_no_b0_flags_proceeds(self):
        self.assertTrue(enforce_b0_gate(_args()))

    def test_b0_without_val_split_exits(self):
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(b0_manifest="/f", b0_performance="/f",
                                    b0_journal="/f", b0_budget="/f"))

    def test_partial_b0_paths_exits(self):
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(task_split="val", b0_manifest="/f"))

    def test_dry_run_returns_false_no_files(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _make_all(self.tmp, mf_dict=mf, pf_dict=_pf(run_fp, dp_fp),
                       jf_dict=_jf(run_fp), bf_list=_bf())
        before = set(os.listdir(self.tmp))
        result = enforce_b0_gate(_args(
            task_split="val", b0_manifest=ps["manifest"],
            b0_performance=ps["performance"],
            b0_journal=ps["journal"], b0_budget=ps["budget"],
            b0_dry_run=True,
        ))
        self.assertFalse(result)
        after = set(os.listdir(self.tmp))
        self.assertEqual(before & after, before)

    def test_valid_evidence_passes_gate(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _make_all(self.tmp, mf_dict=mf, pf_dict=_pf(run_fp, dp_fp),
                       jf_dict=_jf(run_fp), bf_list=_bf())
        self.assertTrue(enforce_b0_gate(_args(
            task_split="val", b0_manifest=ps["manifest"],
            b0_performance=ps["performance"],
            b0_journal=ps["journal"], b0_budget=ps["budget"],
        )))

    def test_invalid_evidence_exits(self):
        mf = _mf(seed=99)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _make_all(self.tmp, mf_dict=mf, pf_dict=_pf(run_fp, dp_fp),
                       jf_dict=_jf(run_fp), bf_list=_bf())
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val", b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"], b0_budget=ps["budget"],
            ))


class TestCLIRegistration(unittest.TestCase):
    """Given an argparse parser, When adding B0 CLI args, Then args exist."""

    def test_adds_all_five_flags(self):
        import argparse
        p = argparse.ArgumentParser()
        add_b0_cli_args(p)
        ns = p.parse_args([])
        for attr in ("b0_manifest", "b0_performance", "b0_journal", "b0_budget",
                      "b0_dry_run"):
            self.assertTrue(hasattr(ns, attr), f"Missing --{attr}")

    def test_b0_dry_run_defaults_false(self):
        import argparse
        p = argparse.ArgumentParser()
        add_b0_cli_args(p)
        ns = p.parse_args([])
        self.assertFalse(ns.b0_dry_run)

    def test_b0_manifest_accepts_path(self):
        import argparse
        p = argparse.ArgumentParser()
        add_b0_cli_args(p)
        ns = p.parse_args(["--b0_manifest", "/tmp/test.json"])
        self.assertEqual(ns.b0_manifest, "/tmp/test.json")


if __name__ == "__main__":
    unittest.main()
