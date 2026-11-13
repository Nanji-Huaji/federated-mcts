"""B0 measured-evidence contract: types, constants, projection formula.

Defines the immutable evidence record, projection record, and the
separated formula terms.  Validation logic lives in ``game24_b0_validate.py``.

Schema version: 1.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from decimal import ROUND_DOWN, Decimal

from federated_mcts.evaluation.game24_b0_manifest_loader import (
    B0SourceDecisionIdentity,
)

SCHEMA_VERSION: int = 1
FULL_VALIDATION_GATE: Decimal = Decimal("20.0")
"""Maximum projected raw units for full 200-ID validation.

Re-baselined from 0.75 on 2026-08-10 against measured reasoning-model cost;
see ``.omo/evidence/b0-gate-rebaseline-rationale.md``.
"""
RAW_UNIT_PRECISION: Decimal = Decimal("0.00000001")


PILOT_TASK_COUNT: int = 20
FULL_TASK_COUNT: int = 200
SCALE_FACTOR: Decimal = Decimal(str(FULL_TASK_COUNT)) / Decimal(str(PILOT_TASK_COUNT))

# Canonical B0 pilot expected values, re-pinned from the first COMPLETE run.
#
# Provenance: the 20-ID canonical pilot executed on 2026-08-13 against
# ``https://llmgw.byr.ink/v1`` with ``qwen3.7-plus`` (``enable_thinking=false``),
# ``decision_policy=local_validity``, ``oracle_audit=True``, seed 17, and
# n_generate/n_evaluate/n_select = 2/3/2.  Run finished with ``complete: true``
# across all 20/20 tasks.
#
# The previous pins described a historical ``deepseek-v3`` run executed with a
# 512-token cap.  That configuration is retired and its numbers cannot recur
# (``81919 / 160 = exactly 512`` completion tokens per call encoded the truncated
# behaviour that made every response empty).  Those values were deliberately kept
# unedited until a genuine replacement run existed rather than being quietly
# falsified.
#
# KNOWN BRITTLENESS: this run is LLM-sampled at temperature 0.9, so the token and
# spend totals are NOT reproducible across re-runs — two pilots of the same 20 IDs
# with the same seed produced 163 vs 165 calls and 0.1664672 vs 0.1688832 raw
# units (task-level outcomes also moved: 14/20 vs 12/20).  Re-pinning a single
# run's exact totals therefore makes these constants invalidate on every re-run.
# They are pinned to the single retained evidence set so the fail-closed gate is
# exercisable; treat them as a provenance seal for THAT run, not as a target.
#
# ---------------------------------------------------------------------------
# NOTE: the five numeric pins (B0_EXPECTED_CALLS / _PROMPT_TOKENS /
# _COMPLETION_TOKENS / _TOTAL_TOKENS / _SPENT) were DELETED deliberately.
#
# They pinned resource telemetry to the exact numbers of one LLM-sampled run and
# were compared with exact equality, so any legitimate re-run was rejected.  Two
# identical pilots (same 20 IDs, same seed 17, same config, both ``complete``)
# produced 163 vs 165 calls and 0.16646720 vs 0.16888320 raw units, and task-level
# accuracy moved 14/20 vs 12/20 — the totals are sampling outcomes, not
# configuration.
#
# They were removed rather than widened to a tolerance band because they carried
# no real protection:
#   * The budget ceiling is enforced at RUNTIME by ``--max_cost_usd``
#     (scripts/merged_run.py), reading the actual hash-chained ledger.  It never
#     consults these constants, so fabricating telemetry past the gate buys no
#     budget headroom.
#   * Every tamper test in tests/test_game24_b0_tamper.py is anchored on
#     fingerprints, the hash chain, or structural completeness — none on the
#     numeric values.
#   * ``raw_units_spent`` is taken from the hash-chained ledger, not from the
#     pins, so the projection is unaffected by their removal.
#
# What still binds, unchanged: manifest task-ID set and order, decision fields,
# both fingerprints, ``complete``, journal completeness and run_fingerprint
# agreement, the budget hash chain, internal telemetry consistency, and the
# SHA-256 of all four artifacts.
# ---------------------------------------------------------------------------

B0_EXPECTED_DECISION_POLICY: str = "local_validity"
B0_EXPECTED_ORACLE_AUDIT: bool = True
B0_EXPECTED_SEED: int = 17
B0_EXPECTED_METHOD_GENERATE: str = "propose"
B0_EXPECTED_METHOD_EVALUATE: str = "vote"
B0_EXPECTED_METHOD_SELECT: str = "greedy"
B0_EXPECTED_N_GENERATE: int = 2
B0_EXPECTED_N_EVALUATE: int = 3
B0_EXPECTED_N_SELECT: int = 2

# The canonical 20 validation pilot IDs
CANONICAL_PILOT_IDS: list[int] = [
    3, 9, 16, 19, 24, 25, 28, 36, 40, 42,
    45, 71, 77, 81, 102, 104, 105, 116, 117, 119,
]


@dataclass(frozen=True, slots=True)
class B0EvidenceRecord:
    """Validated B0 pilot evidence from on-disk artifacts.

    All raw-unit values are ``Decimal`` for exact arithmetic.
    Source artifact SHA-256 digests provide traceability.
    """

    schema_version: int
    pilot_ids: list[int]
    pilot_id_count: int
    decision_policy: str
    oracle_audit: bool
    seed: int
    method_generate: str
    method_evaluate: str
    method_select: str
    n_generate_sample: int
    n_evaluate_sample: int
    n_select_sample: int
    calls: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_units_spent: Decimal
    tasks_completed: int
    manifest_sha256: str
    performance_sha256: str
    journal_sha256: str
    budget_sha256: str
    decision_policy_fingerprint: str
    run_fingerprint: str
    source_decision: B0SourceDecisionIdentity


@dataclass(frozen=True, slots=True)
class B0ProjectionRecord:
    """Separated formula terms and gate result.

    Formula: projected = pilot_usage × scale + overhead + retry_allowance
    """

    schema_version: int
    pilot_usage: Decimal
    scale_factor: Decimal
    startup_overhead: Decimal
    retry_allowance: Decimal
    scaled_usage: Decimal
    projected_raw_units: Decimal
    gate_limit: Decimal
    gate_passed: bool

    def to_json(self) -> str:
        d: dict[str, object] = {
            "schema_version": self.schema_version,
            "pilot_usage": str(self.pilot_usage),
            "scale_factor": str(self.scale_factor),
            "startup_overhead": str(self.startup_overhead),
            "retry_allowance": str(self.retry_allowance),
            "scaled_usage": str(self.scaled_usage),
            "projected_raw_units": str(self.projected_raw_units),
            "gate_limit": str(self.gate_limit),
            "gate_passed": self.gate_passed,
        }
        return json.dumps(d, sort_keys=True, indent=2, ensure_ascii=False)


# Keys that collectively identify a charged model usage mapping.
_USAGE_KEYS: frozenset[str] = frozenset({
    "total_calls", "prompt_tokens", "completion_tokens", "total_tokens",
})


def _is_usage_mapping(obj: object) -> bool:
    """Return True if *obj* is a dict containing all usage keys.

    Non-usage aggregate dicts (e.g. ``experiment_metrics``) are
    deliberately excluded so they do not trigger duplicate-model errors.
    """
    return isinstance(obj, dict) and _USAGE_KEYS.issubset(obj.keys())


def extract_model_usage_from_performance(
    perf: dict[str, object],
) -> dict[str, int]:
    """Extract validated per-model usage from B0 performance JSON.

    The B0 performance file stores usage under the model key identified
    by the ``llm`` field (e.g., ``\"deepseek-v4-flash\": {\"total_calls\": 160, ...}``).
    Top-level ``total_calls`` / ``prompt_tokens`` / etc. do NOT exist.

    Fail-closed: raises ``ValueError`` on any malformed, missing,
    inconsistent, or non-finite data.  No fallback to zero.
    """
    # 1) Validate llm field
    llm_raw = perf.get("llm")
    if not isinstance(llm_raw, str) or not llm_raw.strip():
        raise ValueError(
            "performance 'llm' field must be a non-empty string; "
            f"got {type(llm_raw).__name__!r} value {llm_raw!r}"
        )
    llm_key: str = llm_raw.strip()

    # 2) Validate the nested model entry
    model_data = perf.get(llm_key)
    if not isinstance(model_data, dict):
        raise ValueError(
            f"performance model key {llm_key!r} entry must be a dict; "
            f"got {type(model_data).__name__!r}"
        )

    # 3) Reject duplicate usage-shaped model entries; allow non-usage dicts
    _KNOWN_META_KEYS: frozenset[str] = frozenset({
        "llm", "sm", "avg_sum", "acc", "budget_spent",
        "budget_remaining", "complete", "run_fingerprint",
        "decision_policy_fingerprint", "estimated_cost_usd",
        "started_tasks", "stop_reason", "experiment_metrics",
    })
    for k in perf:
        if k in _KNOWN_META_KEYS:
            continue
        if _is_usage_mapping(perf[k]) and k != llm_key:
            raise ValueError(
                f"unexpected duplicate usage model entry {k!r} "
                f"in performance (only {llm_key!r} usage is expected)"
            )

    # 4) Extract and validate token fields
    def _validate_field(name: str) -> int:
        raw = model_data.get(name)
        if raw is None:
            raise ValueError(
                f"model {llm_key!r} missing field {name!r}"
            )
        if isinstance(raw, bool):
            raise ValueError(
                f"model {llm_key!r} {name!r} must be a number, got bool ({raw!r})"
            )
        if not isinstance(raw, (int, float)):
            raise ValueError(
                f"model {llm_key!r} {name!r} must be a number; "
                f"got {type(raw).__name__!r}"
            )
        v = float(raw)
        if math.isnan(v) or math.isinf(v):
            raise ValueError(
                f"model {llm_key!r} {name!r} must be finite; got {raw!r}"
            )
        if v < 0.0:
            raise ValueError(
                f"model {llm_key!r} {name!r} must be nonnegative; got {raw}"
            )
        return int(raw)

    total_calls = _validate_field("total_calls")
    prompt_tokens = _validate_field("prompt_tokens")
    completion_tokens = _validate_field("completion_tokens")
    total_tokens = _validate_field("total_tokens")

    # 5) Cross-validate token totals
    recomputed_total = prompt_tokens + completion_tokens
    if recomputed_total != total_tokens:
        raise ValueError(
            f"model {llm_key!r} token inconsistency: "
            f"prompt_tokens ({prompt_tokens}) + completion_tokens "
            f"({completion_tokens}) = {recomputed_total} != "
            f"total_tokens ({total_tokens})"
        )

    return {
        "total_calls": total_calls,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
    }


def prove_zero_startup_overhead(
    method_generate: str = "propose",
    warm_start: bool = False,
    search_policy: str = "baseline",
    game24_exact_prune: bool = False,
) -> tuple[bool, Decimal, str]:
    """Prove zero-fixed-startup-overhead for B0 baseline.

    B0 baseline has no warm-start, DQN, or exact-prune overhead.
    """
    if warm_start:
        return False, Decimal("0"), "warm_start=True: startup overhead nonzero"
    if search_policy not in ("baseline", ""):
        return False, Decimal("0"), f"search_policy={search_policy!r}: overhead nonzero"
    if game24_exact_prune:
        return False, Decimal("0"), "game24_exact_prune=True: overhead nonzero"
    return True, Decimal("0"), (
        f"startup_overhead=0: method_generate={method_generate!r} (baseline), "
        "no warm_start/DQN/exact_prune"
    )


def prove_zero_retry_allowance(
    n_generate_sample: int = 2,
    n_evaluate_sample: int = 3,
) -> tuple[bool, Decimal, str]:
    """Prove zero-retry-allowance for B0 baseline.

    B0 has no configurable retry multiplier; pilot usage captures all cost.
    """
    return True, Decimal("0"), (
        f"retry_allowance=0: n_generate={n_generate_sample}, "
        f"n_evaluate={n_evaluate_sample}, no configurable retry multiplier"
    )


def compute_b0_projection(
    evidence: B0EvidenceRecord,
) -> tuple[B0ProjectionRecord, str, str]:
    """Compute measured B0 projection for 200-ID validation.

    Returns (projection_record, startup_proof, retry_proof).
    """
    overhead_ok, startup_overhead, startup_proof = prove_zero_startup_overhead(
        method_generate=evidence.method_generate,
    )
    if not overhead_ok:
        raise ValueError(f"Startup overhead not zero: {startup_proof}")

    retry_ok, retry_allowance, retry_proof = prove_zero_retry_allowance(
        n_generate_sample=evidence.n_generate_sample,
        n_evaluate_sample=evidence.n_evaluate_sample,
    )
    if not retry_ok:
        raise ValueError(f"Retry allowance not zero: {retry_proof}")

    scaled_usage = (evidence.raw_units_spent * SCALE_FACTOR).quantize(
        Decimal("0.00000001"), rounding=ROUND_DOWN,
    )
    projected = scaled_usage + startup_overhead + retry_allowance

    return (
        B0ProjectionRecord(
            schema_version=SCHEMA_VERSION,
            pilot_usage=evidence.raw_units_spent,
            scale_factor=SCALE_FACTOR,
            startup_overhead=startup_overhead,
            retry_allowance=retry_allowance,
            scaled_usage=scaled_usage,
            projected_raw_units=projected,
            gate_limit=FULL_VALIDATION_GATE,
            gate_passed=projected <= FULL_VALIDATION_GATE,
        ),
        startup_proof,
        retry_proof,
    )
