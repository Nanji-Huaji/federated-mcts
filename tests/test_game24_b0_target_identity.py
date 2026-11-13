"""Tests for B0 gate target-identity enforcement: valid match and drift rejection."""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_gate import enforce_b0_gate
from federated_mcts.evaluation.game24_manifest import Game24Manifest

from tests._game24_b0_fixtures import (
    _args,
    _bf,
    _derive_fps,
    _jf,
    _make_all,
    _mf,
    _pf,
)


def _build_target_manifest(*, task_ids=None, **overrides) -> Game24Manifest:
    """Build a Game24Manifest matching the B0 source fixture (& overrides).

    Uses a minimal model_config_raw and temp split file so __post_init__
    produces real hashes; then overrides model_config_sha256 and
    model_names to match the B0 source fixture values.
    """
    import tempfile as _tmp

    ids = task_ids if task_ids is not None else list(range(3, 1203))
    # Create a minimal split file for __post_init__
    split_file = _tmp.NamedTemporaryFile(
        mode="w", suffix=".json", delete=False,
    )
    split_file.write(json.dumps({"val": ids}))
    split_file.close()
    try:
        m = Game24Manifest(
            task="game24",
            task_ids=ids,
            task_split_name="val",
            decision_policy="local_validity",
            oracle_audit=True,
            seed=17,
            temperature=0.9,
            method_generate="propose",
            method_evaluate="vote",
            method_select="greedy",
            n_generate_sample=2,
            n_evaluate_sample=3,
            n_select_sample=2,
            game24_exact_prune=False,
            assign_strategy="round_robin",
            max_call_limit=None,
            max_token_limit=None,
            max_cost_usd=None,
            max_raw_unit_limit=19.99,
            split_file_path=split_file.name,
            model_config_raw=[{"model": "qwen3.7-plus"}],
        )
    finally:
        os.unlink(split_file.name)
    # Override computed fields to match B0 source fixture
    object.__setattr__(m, "model_config_sha256", "c" * 64)
    object.__setattr__(m, "model_names", ["qwen3.7-plus"])
    # Allow caller overrides
    for k, v in overrides.items():
        object.__setattr__(m, k, v)
    return m


def _val_200_ids() -> list[int]:
    """200 IDs for full validation (non-pilot range, representative)."""
    import json as _j
    # Use the real split file if available; fall back to synthetic 200
    split_path = "data/splits/game24.json"
    if os.path.exists(split_path):
        with open(split_path) as f:
            return sorted(_j.load(f)["val"])
    return sorted(set(range(3, 1368)))[:200]


class TestTargetIdentityGate(unittest.TestCase):
    """Given a target manifest, When enforcing B0 gate with identity check,
    Then matching identity passes, any drift rejects fail-closed."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _b0_fixture_paths(self, mf_dict=None, **overrides) -> dict[str, str]:
        """Create B0 evidence fixtures and return path dict."""
        mf = dict(_mf())
        if mf_dict is not None:
            mf.update(mf_dict)
        mf.update(overrides)
        dp_fp, run_fp = _derive_fps(mf)
        return _make_all(self.tmp,
                         mf_dict=mf,
                         pf_dict=_pf(run_fp, dp_fp),
                         jf_dict=_jf(run_fp),
                         bf_list=_bf())

    # ── Happy path: target matches B0 source ──────────────────────

    def test_target_matches_b0_source_passes(self):
        """200 val target with same decision fields as B0 source passes."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(task_ids=_val_200_ids())
        result = enforce_b0_gate(_args(
            task_split="val",
            b0_manifest=ps["manifest"],
            b0_performance=ps["performance"],
            b0_journal=ps["journal"],
            b0_budget=ps["budget"],
        ), manifest=target)
        self.assertTrue(result)

    def test_target_matches_b0_source_dry_run_passes(self):
        """Identity gate passes on dry-run without side effects."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(task_ids=_val_200_ids())
        before = set(os.listdir(self.tmp))
        result = enforce_b0_gate(_args(
            task_split="val",
            b0_manifest=ps["manifest"],
            b0_performance=ps["performance"],
            b0_journal=ps["journal"],
            b0_budget=ps["budget"],
            b0_dry_run=True,
        ), manifest=target)
        self.assertFalse(result)
        after = set(os.listdir(self.tmp))
        self.assertEqual(before & after, before)

    # ── Model drift rejection ─────────────────────────────────────

    def test_model_config_hash_drift_rejects(self):
        """Model config SHA-256 mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            model_config_sha256="d" * 64,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_model_names_drift_rejects(self):
        """Model names change rejects fail-closed (e.g. Qwen 7B vs 32B)."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            model_names=["Qwen/Qwen2.5-7B-Instruct", "qwen3.7-plus"],
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    # ── Decision setting drift rejection ──────────────────────────

    def test_seed_drift_rejects(self):
        """Seed mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            seed=42,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_temperature_drift_rejects(self):
        """Temperature mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            temperature=0.3,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_n_generate_drift_rejects(self):
        """n_generate_sample mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            n_generate_sample=1,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_exact_prune_drift_rejects(self):
        """game24_exact_prune mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            game24_exact_prune=True,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_raw_unit_limit_drift_rejects(self):
        """max_raw_unit_limit mismatch rejects fail-closed."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(
            task_ids=_val_200_ids(),
            max_raw_unit_limit=10.0,
        )
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    # ── Task identity violations ──────────────────────────────────

    def test_non_game24_task_rejects(self):
        """Target task not 'game24' rejects."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(task_ids=_val_200_ids())
        object.__setattr__(target, "task", "crosswords")
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)

    def test_not_200_val_ids_rejects(self):
        """Target with non-200 task_ids rejects."""
        ps = self._b0_fixture_paths()
        target = _build_target_manifest(task_ids=list(range(3, 23)))  # 20 IDs
        with self.assertRaises(SystemExit):
            enforce_b0_gate(_args(
                task_split="val",
                b0_manifest=ps["manifest"],
                b0_performance=ps["performance"],
                b0_journal=ps["journal"],
                b0_budget=ps["budget"],
            ), manifest=target)


if __name__ == "__main__":
    unittest.main()
