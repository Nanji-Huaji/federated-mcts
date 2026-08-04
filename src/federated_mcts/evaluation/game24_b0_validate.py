"""B0 evidence validation: load, verify, and hash-check pilot artifacts.

Validates all four B0 evidence files (manifest, performance, journal,
budget) against canonical expectations.  Raises ``ValueError`` on
any discrepancy (fail-closed).  Returns a validated ``B0EvidenceRecord``.

Fingerprints are derived from manifest fields using the same canonical
JSON hashing as ``Game24Manifest`` — the serialized manifest JSON may
omit the derived properties and this validator reconstructs them.

Does NOT perform mutation, file creation, network I/O, or model construction.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from decimal import ROUND_DOWN, Decimal

from federated_mcts.evaluation.game24_b0_evidence import (
    B0EvidenceRecord,
    B0_EXPECTED_CALLS,
    B0_EXPECTED_COMPLETION_TOKENS,
    B0_EXPECTED_DECISION_POLICY,
    B0_EXPECTED_METHOD_EVALUATE,
    B0_EXPECTED_METHOD_GENERATE,
    B0_EXPECTED_METHOD_SELECT,
    B0_EXPECTED_N_EVALUATE,
    B0_EXPECTED_N_GENERATE,
    B0_EXPECTED_N_SELECT,
    B0_EXPECTED_ORACLE_AUDIT,
    B0_EXPECTED_PROMPT_TOKENS,
    B0_EXPECTED_SEED,
    B0_EXPECTED_TOTAL_TOKENS,
    CANONICAL_PILOT_IDS,
    PILOT_TASK_COUNT,
    RAW_UNIT_PRECISION,
    SCHEMA_VERSION,
)
from federated_mcts.evaluation.game24_b0_manifest_loader import (
    derive_fingerprints_from_manifest_dict,
    extract_source_decision_identity,
)
from federated_mcts.evaluation.game24_b0_verify import (
    verify_journal,
    verify_budget_ledger,
)


def _sha256_file(path: str) -> str:
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


def _check_finite_nonnegative(name: str, value: object) -> int:
    """Validate *value* is finite, nonnegative, not bool; return as int.

    Raises ``ValueError`` on NaN, Inf, negative, or bool.
    """
    if isinstance(value, bool):
        raise ValueError(f"{name} must be a number, got bool ({value!r})")
    if not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {type(value).__name__}")
    v = float(value)
    if math.isnan(v) or math.isinf(v):
        raise ValueError(f"{name} must be finite, got {value!r}")
    if v < 0.0:
        raise ValueError(f"{name} must be nonnegative, got {value}")
    return int(value)


def _to_int(value: object, default: int = 0) -> int:
    """Coerce value to int; returns default for incompatible types."""
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (str, float)):
        return int(value)
    return default


def _load_json(path: str, label: str) -> dict[str, object]:
    try:
        with open(path) as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise ValueError(f"{label} is not valid JSON: {e}")
    except OSError as e:
        raise ValueError(f"Cannot read {label}: {e}")
    if not isinstance(data, dict):
        raise ValueError(f"{label} top-level must be object, got {type(data).__name__}")
    return data


def _verify_manifest_settings(mf: dict[str, object]) -> None:
    checks: list[tuple[str, object, object]] = [
        ("decision_policy", mf.get("decision_policy"), B0_EXPECTED_DECISION_POLICY),
        ("oracle_audit", mf.get("oracle_audit"), B0_EXPECTED_ORACLE_AUDIT),
        ("seed", mf.get("seed"), B0_EXPECTED_SEED),
        ("method_generate", mf.get("method_generate"), B0_EXPECTED_METHOD_GENERATE),
        ("method_evaluate", mf.get("method_evaluate"), B0_EXPECTED_METHOD_EVALUATE),
        ("method_select", mf.get("method_select"), B0_EXPECTED_METHOD_SELECT),
        ("n_generate_sample", mf.get("n_generate_sample"), B0_EXPECTED_N_GENERATE),
        ("n_evaluate_sample", mf.get("n_evaluate_sample"), B0_EXPECTED_N_EVALUATE),
        ("n_select_sample", mf.get("n_select_sample"), B0_EXPECTED_N_SELECT),
    ]
    for name, actual, expected in checks:
        if actual != expected:
            raise ValueError(
                f"B0 manifest {name!r}: expected {expected!r}, got {actual!r}"
            )
    task_ids = mf.get("task_ids", [])
    if not isinstance(task_ids, list):
        raise ValueError("B0 manifest task_ids is not a list")
    if len(task_ids) != PILOT_TASK_COUNT:
        raise ValueError(
            f"B0 manifest task_ids count: expected {PILOT_TASK_COUNT}, got {len(task_ids)}"
        )
    if task_ids != CANONICAL_PILOT_IDS:
        raise ValueError(
            f"B0 manifest task_ids order mismatch. "
            f"Expected {CANONICAL_PILOT_IDS}, got {task_ids}"
        )


def _verify_performance(perf: dict[str, object]) -> dict[str, int]:
    """Validate B0 performance telemetry against canonical expectations.

    Returns the extracted usage dict for downstream record construction.
    Raises ``ValueError`` on any mismatch (fail-closed).
    """
    from federated_mcts.evaluation.game24_b0_evidence import (
        extract_model_usage_from_performance,
    )

    usage = extract_model_usage_from_performance(perf)

    for field, expected in [
        ("total_calls", B0_EXPECTED_CALLS),
        ("prompt_tokens", B0_EXPECTED_PROMPT_TOKENS),
        ("completion_tokens", B0_EXPECTED_COMPLETION_TOKENS),
        ("total_tokens", B0_EXPECTED_TOTAL_TOKENS),
    ]:
        value = usage[field]
        if value != expected:
            raise ValueError(
                f"Performance {field}: expected {expected}, got {value}"
            )

    return usage


def _validate_paths(paths: dict[str, str]) -> None:
    for label, path in paths.items():
        if not path:
            raise ValueError(f"B0 {label} path not provided")
        if not os.path.isfile(path):
            raise ValueError(f"B0 {label} not found or not a regular file: {path}")


def validate_b0_evidence(
    *,
    manifest_path: str,
    performance_path: str,
    journal_path: str,
    budget_path: str,
) -> B0EvidenceRecord:
    """Validate all four B0 pilot evidence artifacts.

    Strict validation steps (fail-closed on any failure):
    1. File existence check
    2. JSON/JSONL parse validation
    3. Manifest settings (canonical IDs, policy, seed, config)
    4. Fingerprint derivation from manifest fields (canonical reconstruction)
    5. Performance telemetry (calls, tokens match B0 pilot)
    6. Journal completeness + run_fingerprint match
    7. Budget hash-chain verification + spent reconciliation
    8. Cross-verify performance budget_spent vs ledger spent
    9. SHA-256 all source artifacts
    10. Finite nonnegative value checks throughout

    Returns validated ``B0EvidenceRecord``.  Raises ``ValueError`` on
    any failure.  Zero side effects.
    """
    paths = {
        "manifest": manifest_path, "performance": performance_path,
        "journal": journal_path, "budget": budget_path,
    }
    _validate_paths(paths)

    manifest_sha = _sha256_file(manifest_path)
    performance_sha = _sha256_file(performance_path)
    journal_sha = _sha256_file(journal_path)
    budget_sha = _sha256_file(budget_path)

    mf = _load_json(manifest_path, "B0 manifest")
    _verify_manifest_settings(mf)

    # Derive fingerprints from canonical manifest fields (not from JSON)
    dp_fp, run_fp = derive_fingerprints_from_manifest_dict(mf)

    # Extract source decision identity for target-identity gate
    source_decision = extract_source_decision_identity(mf)

    perf = _load_json(performance_path, "B0 performance")
    usage = _verify_performance(perf)

    budget_spent_raw = perf.get("budget_spent", 0)
    _check_finite_nonnegative("budget_spent", budget_spent_raw)
    budget_spent_perf = Decimal(str(budget_spent_raw))

    perf_fp = str(perf.get("run_fingerprint", ""))
    if not perf_fp:
        raise ValueError("B0 performance missing run_fingerprint field")
    if perf_fp != run_fp:
        raise ValueError(
            f"B0 performance run_fingerprint {perf_fp[:20]}... "
            f"!= manifest derived {run_fp[:20]}..."
        )

    completed_ids = verify_journal(journal_path, run_fp)
    budget_spent_ledger = verify_budget_ledger(budget_path)

    if budget_spent_perf.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN) != \
       budget_spent_ledger.quantize(RAW_UNIT_PRECISION, rounding=ROUND_DOWN):
        raise ValueError(
            f"B0 budget spent mismatch: "
            f"performance={budget_spent_perf}, ledger={budget_spent_ledger}"
        )

    return B0EvidenceRecord(
        schema_version=SCHEMA_VERSION,
        pilot_ids=list(CANONICAL_PILOT_IDS),
        pilot_id_count=PILOT_TASK_COUNT,
        decision_policy=str(mf.get("decision_policy", "")),
        oracle_audit=bool(mf.get("oracle_audit", False)),
        seed=_to_int(mf.get("seed", -1)),
        method_generate=str(mf.get("method_generate", "")),
        method_evaluate=str(mf.get("method_evaluate", "")),
        method_select=str(mf.get("method_select", "")),
        n_generate_sample=_to_int(mf.get("n_generate_sample", 0)),
        n_evaluate_sample=_to_int(mf.get("n_evaluate_sample", 0)),
        n_select_sample=_to_int(mf.get("n_select_sample", 0)),
        calls=usage["total_calls"],
        prompt_tokens=usage["prompt_tokens"],
        completion_tokens=usage["completion_tokens"],
        total_tokens=usage["total_tokens"],
        raw_units_spent=budget_spent_ledger,
        tasks_completed=len(completed_ids),
        manifest_sha256=manifest_sha,
        performance_sha256=performance_sha,
        journal_sha256=journal_sha,
        budget_sha256=budget_sha,
        decision_policy_fingerprint=dp_fp,
        run_fingerprint=run_fp,
        source_decision=source_decision,
    )
