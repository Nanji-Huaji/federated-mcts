"""Tests for B0 evidence validation.

Covers valid, corrupt, missing, incomplete, mismatched, reordered,
nonfinite evidence, no-side-effect guarantee, and regression for
canonical manifest fingerprint derivation (is stable from derived fields).

All fixtures are synthetic.  ``decision_policy_fingerprint`` and
``run_fingerprint`` are NOT placed in the manifest JSON — they
are derived by the validator using the same canonical hashing as
``Game24Manifest``.

Tamper/mismatch tests live in ``test_game24_b0_tamper.py``.
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from decimal import Decimal

sys.path.insert(0, "src")

from federated_mcts.evaluation.game24_b0_evidence import (
    B0_EXPECTED_SPENT,
    CANONICAL_PILOT_IDS,
    RAW_UNIT_PRECISION,
    extract_model_usage_from_performance,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    derive_fingerprints_from_manifest_dict,
)
from federated_mcts.evaluation.game24_b0_validate import validate_b0_evidence


def _mf(**kw) -> dict:
    """Build B0 manifest dict with canonical B0 field values.

    Includes ALL fields from ``_DECISION_FIELDS`` and ``_RUN_FIELDS``
    so that ``_dict_to_canonical_subset`` can validate completeness
    (fail-closed on missing required fields).
    """
    d: dict = {
        "task": "game24",
        "task_ids": list(CANONICAL_PILOT_IDS),
        "task_split_name": "val",
        "decision_policy": "local_validity",
        "oracle_audit": True,
        "seed": 17,
        "temperature": 0.9,
        "method_generate": "propose",
        "method_evaluate": "vote",
        "method_select": "greedy",
        "n_generate_sample": 2,
        "n_evaluate_sample": 3,
        "n_select_sample": 2,
        "game24_exact_prune": False,
        # ── Budget limits ─────────────────────────────────
        "max_call_limit": None,
        "max_token_limit": None,
        "max_cost_usd": None,
        "max_raw_unit_limit": 19.99,
        # ── Solver extras ─────────────────────────────────
        "assign_strategy": "round_robin",
        "runner_path": "scripts/merged_run.py",
        "effective_cli": "",
        # ── File / model / git hashes (placeholder) ────────
        "split_file_sha256": "a" * 64,
        "ordered_task_ids_sha256": "b" * 64,
        "model_config_sha256": "c" * 64,
        "model_names": ["deepseek-v4-flash"],
        # ── Git metadata ──────────────────────────────────
        "git_commit": "0000000",
        "worktree_dirty": False,
    }
    d.update(kw)
    return d


def _derive_fps(mf_dict: dict) -> tuple[str, str]:
    return derive_fingerprints_from_manifest_dict(mf_dict)


def _pf(run_fp: str, dp_fp: str, **kw) -> dict:
    d: dict = {
        "acc": 0.0, "avg_sum": 0.0,
        "llm": "deepseek-v4-flash",
        "deepseek-v4-flash": {
            "total_calls": 160, "prompt_tokens": 64960,
            "completion_tokens": 81919, "total_tokens": 146879,
            "cost": 0.03203172,
        },
        "estimated_cost_usd": 0.03203172,
        "budget_spent": 0.03203172,
        "budget_remaining": 19.95796828,
        "complete": True, "run_fingerprint": run_fp,
        "decision_policy_fingerprint": dp_fp,
    }
    d.update(kw)
    return d


def _jf(run_fp: str, task_ids: list[int] | None = None) -> dict:
    ids = task_ids if task_ids is not None else list(CANONICAL_PILOT_IDS)
    j: dict = {"_meta": {"run_fingerprint": run_fp}}
    for tid in ids:
        j[str(tid)] = {"task_id": tid, "ys": [f"r{tid}"], "_completed": True}
    return j


def _bf(spent=0.03203172, limit=19.99) -> list[dict]:
    import hashlib

    def sh(obj):
        d = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                       ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(d).hexdigest()

    init = {"kind": "init", "timestamp_utc": "2025-01-01T00:00:00Z",
            "prev_hash": "", "total_limit": limit}
    init["entry_hash"] = sh(init)
    se = {"kind": "spend", "timestamp_utc": "2025-01-01T00:01:00Z",
          "prev_hash": init["entry_hash"], "amount": spent,
          "remaining": limit - spent}
    se["entry_hash"] = sh(se)
    return [init, se]


def _wr(tmp, mf=None, pf=None, jf=None, bf=None) -> dict[str, str]:
    ps: dict[str, str] = {}
    for label, data, ext in [
        ("manifest", mf, "json"), ("performance", pf, "json"),
        ("journal", jf, "json"), ("budget", bf, "jsonl"),
    ]:
        if data is None:
            continue
        p = os.path.join(tmp, f"{label}.{ext}")
        with open(p, "w") as f:
            if ext == "jsonl":
                for e in data:
                    f.write(json.dumps(e, sort_keys=True,
                                       separators=(",", ":")) + "\n")
            else:
                json.dump(data, f)
        ps[label] = p
    return ps


class TestB0ValidEvidence(unittest.TestCase):
    """Given correct B0 evidence, When validating, Then passes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_valid_evidence_passes_all_checks(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        ev = validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertEqual(ev.pilot_id_count, 20)
        self.assertEqual(ev.raw_units_spent, B0_EXPECTED_SPENT)
        self.assertTrue(ev.manifest_sha256)

    def test_validation_creates_no_files(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        before = set(os.listdir(self.tmp))
        validate_b0_evidence(**{f"{k}_path": v for k, v in ps.items()})
        self.assertEqual(before, set(os.listdir(self.tmp)))

    def test_fingerprints_derived_from_canonical_fields(self):
        """When manifest has no fp fields, validator derives correct fingerprints."""
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        self.assertEqual(len(dp_fp), 64)
        self.assertEqual(len(run_fp), 64)
        self.assertNotEqual(dp_fp, run_fp)
        dp_fp2, run_fp2 = _derive_fps(_mf())
        self.assertEqual(dp_fp, dp_fp2)
        self.assertEqual(run_fp, run_fp2)

    def test_null_budget_limit_produces_stable_fingerprint(self):
        """Null max_call_limit is included in canonical hash, stable across calls."""
        mf = _mf()
        self.assertIsNone(mf["max_call_limit"])
        dp1, rn1 = _derive_fps(mf)
        dp2, rn2 = _derive_fps(_mf())  # same manifest = same fp
        self.assertEqual(dp1, dp2)
        self.assertEqual(rn1, rn2)

    def test_null_to_nonnull_changes_fingerprint(self):
        """Changing max_call_limit from None to a value changes fingerprints."""
        mf_null = _mf()
        mf_val = _mf(max_call_limit=100)
        dp_null, rn_null = _derive_fps(mf_null)
        dp_val, rn_val = _derive_fps(mf_val)
        self.assertNotEqual(dp_null, dp_val, "decision fp unchanged after null→100")
        self.assertNotEqual(rn_null, rn_val, "run fp unchanged after null→100")

    def test_null_budget_limit_present_in_canonical_like_live_manifest(self):
        """Null budget limits are in canonical dict just as Game24Manifest includes them."""
        from federated_mcts.evaluation.game24_manifest import (
            _DECISION_FIELDS,
            _canonical_json_bytes,
        )
        from federated_mcts.evaluation.game24_b0_manifest_loader import (
            _dict_to_canonical_subset,
        )
        mf = _mf()
        # Decision dict must contain the null budget fields
        d = _dict_to_canonical_subset(mf, _DECISION_FIELDS)
        self.assertIn("max_call_limit", d)
        self.assertIsNone(d["max_call_limit"])
        self.assertIn("max_token_limit", d)
        self.assertIsNone(d["max_token_limit"])
        self.assertIn("max_cost_usd", d)
        self.assertIsNone(d["max_cost_usd"])
        # Canonical JSON must include "null" for those keys
        raw = _canonical_json_bytes(d).decode()
        self.assertIn('"max_call_limit":null', raw)
        self.assertIn('"max_token_limit":null', raw)
        self.assertIn('"max_cost_usd":null', raw)


class TestB0Errors(unittest.TestCase):
    """Given various invalid evidence, When validating, Then ValueError."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _v(self, **kw) -> None:
        all_kw: dict[str, str] = {}
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        defaults = [
            ("manifest_path", os.path.join(self.tmp, "_dm.json"), mf, "w"),
            ("performance_path", os.path.join(self.tmp, "_dp.json"),
             _pf(run_fp, dp_fp), "w"),
            ("journal_path", os.path.join(self.tmp, "_dj.json"),
             _jf(run_fp), "w"),
            ("budget_path", os.path.join(self.tmp, "_db.jsonl"),
             _bf(), "w"),
        ]
        for key, path, data, _mode in defaults:
            if key not in kw:
                with open(path, _mode) as f:
                    if isinstance(data, list):
                        for e in data:
                            f.write(json.dumps(e, sort_keys=True,
                                               separators=(",", ":")) + "\n")
                    else:
                        json.dump(data, f)
                all_kw[key] = path
        all_kw.update(kw)
        validate_b0_evidence(**all_kw)

    # — missing files —
    def test_missing_manifest(self):
        ps = _wr(self.tmp, mf=_mf(), pf=_pf("a" * 64, "b" * 64),
                 jf=_jf("a" * 64), bf=_bf())
        os.remove(ps["manifest"])
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — corrupt JSON —
    def test_corrupt_json(self):
        p = os.path.join(self.tmp, "manifest.json")
        with open(p, "w") as f:
            f.write("not json")
        with self.assertRaises(ValueError):
            self._v(manifest_path=p)

    # — wrong settings —
    def test_wrong_decision_policy(self):
        mf = _mf(decision_policy="heuristic_rules")
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    def test_wrong_seed(self):
        mf = _mf(seed=42)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — wrong task IDs —
    def test_truncated_task_ids(self):
        short = CANONICAL_PILOT_IDS[:19]
        mf = _mf(task_ids=short)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp, task_ids=short), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    def test_reordered_task_ids(self):
        rev = list(reversed(CANONICAL_PILOT_IDS))
        mf = _mf(task_ids=rev)
        dp_fp, run_fp = _derive_fps(mf)
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp),
                 jf=_jf(run_fp, task_ids=rev), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(manifest_path=ps["manifest"])

    # — incomplete journal —
    def test_incomplete_journal(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        j = _jf(run_fp)
        j[str(CANONICAL_PILOT_IDS[-1])]["_completed"] = False
        ps = _wr(self.tmp, mf=mf, pf=_pf(run_fp, dp_fp), jf=j, bf=_bf())
        with self.assertRaises(ValueError):
            self._v(journal_path=ps["journal"])

    # — nonfinite values —
    def test_nan(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"]["total_calls"] = float("nan")
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_inf(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"]["total_tokens"] = float("inf")
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_negative(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"]["total_calls"] = -5
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    # ── Nested usage shape edge cases ─────────────────────────────

    def test_missing_llm_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["llm"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_llm_field_not_string_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["llm"] = 42
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_missing_model_usage_entry_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["deepseek-v4-flash"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_model_entry_not_dict_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"] = "not a dict"
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_multiple_model_entries_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-pro"] = {"total_calls": 5, "prompt_tokens": 100,
                                  "completion_tokens": 200, "total_tokens": 300}
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_token_total_inconsistency_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"]["total_tokens"] = 999
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_missing_nested_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        del pf["deepseek-v4-flash"]["completion_tokens"]
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    def test_bool_nested_field_fails(self):
        mf = _mf()
        dp_fp, run_fp = _derive_fps(mf)
        pf = _pf(run_fp, dp_fp)
        pf["deepseek-v4-flash"]["prompt_tokens"] = True
        ps = _wr(self.tmp, mf=mf, pf=pf,
                 jf=_jf(run_fp), bf=_bf())
        with self.assertRaises(ValueError):
            self._v(performance_path=ps["performance"])

    # ── Missing required fingerprint field → fail-closed ───────────

    def test_missing_max_call_limit_rejected(self):
        """When max_call_limit is absent from manifest, fingerprint derivation fails."""
        mf = _mf()
        del mf["max_call_limit"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_call_limit", str(ctx.exception))

    def test_missing_max_token_limit_rejected(self):
        mf = _mf()
        del mf["max_token_limit"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_token_limit", str(ctx.exception))

    def test_missing_max_cost_usd_rejected(self):
        mf = _mf()
        del mf["max_cost_usd"]
        with self.assertRaises(ValueError) as ctx:
            _derive_fps(mf)
        self.assertIn("max_cost_usd", str(ctx.exception))


# ── Unit tests for extract_model_usage_from_performance ─────────

class TestExtractModelUsageFromPerformance(unittest.TestCase):
    """Given various performance dict shapes, When extract_model_usage_from_performance
       is called, Then usage is extracted or ValueError is raised."""

    def test_experiment_metrics_coexists_passes(self):
        """experiment_metrics (non-usage dict) coexists with model usage — passes."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {
                "version": 2, "task_count": 20, "solved_task_count": 8,
                "success_rate": 0.4, "total_proposed": 500,
                "total_candidates": 500, "rejection_counts": {},
            },
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)
        self.assertEqual(usage["total_tokens"], 146879)

    def test_experiment_metrics_alone_does_not_trigger_duplicate_error(self):
        """experiment_metrics keys (task_count etc.) are not usage keys — no false positive."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {"task_count": 20, "solved_task_count": 8},
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)

    def test_duplicate_usage_mapping_rejected(self):
        """A second dict with all usage keys is still rejected."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "deepseek-v4-pro": {
                "total_calls": 5, "prompt_tokens": 100,
                "completion_tokens": 200, "total_tokens": 300,
            },
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_duplicate_usage_mapping_rejected_with_experiment_metrics(self):
        """A second usage mapping is rejected even when experiment_metrics is also present."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "experiment_metrics": {"task_count": 20},
            "deepseek-v4-pro": {
                "total_calls": 5, "prompt_tokens": 100,
                "completion_tokens": 200, "total_tokens": 300,
            },
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_malformed_primary_mapping_still_fails(self):
        """Malformed primary model entry (missing field) still raises ValueError."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919,
                # total_tokens missing
            },
            "experiment_metrics": {"task_count": 20},
        }
        with self.assertRaises(ValueError):
            extract_model_usage_from_performance(perf)

    def test_arbitrary_non_usage_dict_ignored(self):
        """An unknown dict without usage keys is silently allowed — no false rejection."""
        perf = {
            "llm": "deepseek-v4-flash",
            "deepseek-v4-flash": {
                "total_calls": 160, "prompt_tokens": 64960,
                "completion_tokens": 81919, "total_tokens": 146879,
            },
            "some_config": {"timeout": 30, "retries": 3},
        }
        usage = extract_model_usage_from_performance(perf)
        self.assertEqual(usage["total_calls"], 160)


# ── Tests for RAW_UNIT_PRECISION quantization — boundary and binary-tail ─

class TestRawUnitPrecisionBoundary(unittest.TestCase):
    """Given raw-unit Decimals, When quantizing to RAW_UNIT_PRECISION (ROUND_DOWN),
    Then binary-float tails are absorbed but materially-different values reject."""

    def _q(self, value):
        from decimal import ROUND_DOWN
        return value.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN)

    def test_exact_value_quantizes_to_self(self):
        result = self._q(B0_EXPECTED_SPENT)
        self.assertEqual(result, B0_EXPECTED_SPENT.quantize(RAW_UNIT_PRECISION))

    def test_binary_tail_passes(self):
        noisy = Decimal("0.032031720000000014")
        canonical = self._q(B0_EXPECTED_SPENT)
        self.assertEqual(self._q(noisy), canonical)

    def test_binary_tail_greater_passes(self):
        noisy = Decimal("0.03203172000000001")
        canonical = self._q(B0_EXPECTED_SPENT)
        self.assertEqual(self._q(noisy), canonical)

    def test_one_unit_less_rejects(self):
        wrong = Decimal("0.03203171")
        canonical = self._q(B0_EXPECTED_SPENT)
        self.assertNotEqual(self._q(wrong), canonical)

    def test_one_unit_more_rejects(self):
        wrong = Decimal("0.03203173")
        canonical = self._q(B0_EXPECTED_SPENT)
        self.assertNotEqual(self._q(wrong), canonical)


# ── End-to-end: budget ledger with binary-float noise ────────────

class TestBudgetLedgerBinaryTail(unittest.TestCase):
    """Given a budget JSONL file containing the canonical B0 spend,
    When BudgetLedger replays it and returns the float spent value,
    Then verify_budget_ledger normalizes the binary-float tail and passes."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()

    def tearDown(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _write_budget(self, spent_amount=0.03203172, limit=19.99) -> str:
        import hashlib

        def sh(obj):
            d = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                           ensure_ascii=False).encode("utf-8")
            return hashlib.sha256(d).hexdigest()

        init = {"kind": "init", "timestamp_utc": "2025-01-01T00:00:00Z",
                "prev_hash": "", "total_limit": limit}
        init["entry_hash"] = sh(init)
        se = {"kind": "spend", "timestamp_utc": "2025-01-01T00:01:00Z",
              "prev_hash": init["entry_hash"], "amount": spent_amount,
              "remaining": limit - spent_amount}
        se["entry_hash"] = sh(se)

        budget_path = os.path.join(self.tmp, "budget.jsonl")
        with open(budget_path, "w") as f:
            f.write(json.dumps(init, sort_keys=True, separators=(",", ":")) + "\n")
            f.write(json.dumps(se, sort_keys=True, separators=(",", ":")) + "\n")
        return budget_path

    def test_binary_tail_ledger_verifies(self):
        """verify_budget_ledger accepts binary-tail float from ledger."""
        from decimal import ROUND_DOWN
        from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger

        budget_path = self._write_budget()
        spent = verify_budget_ledger(budget_path)
        expected = B0_EXPECTED_SPENT.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN)
        self.assertEqual(spent, expected)

    def test_materially_different_amount_rejects(self):
        """A budget with 0.03203173 must reject even after quantization."""
        from federated_mcts.evaluation.game24_b0_verify import verify_budget_ledger

        budget_path = self._write_budget(spent_amount=0.03203173)
        with self.assertRaises(ValueError):
            verify_budget_ledger(budget_path)


if __name__ == "__main__":
    unittest.main()
