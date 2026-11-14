"""Typed B0 manifest loading with canonical fingerprint derivation.

Loads a serialized Game24 manifest dict (from ``_manifest.json``) and
derives ``decision_policy_fingerprint`` and ``run_fingerprint`` using
the same deterministic canonical-JSON hashing as ``Game24Manifest``.

The corrected B0 idx=1 manifest JSON intentionally omits the derived
fingerprint fields. This module reconstructs them from the fields
that ARE present: ``_DECISION_FIELDS`` and ``_RUN_FIELDS`` as defined
in ``game24_manifest.py``.

Also provides ``B0SourceDecisionIdentity`` — the subset of decision
fields from the validated B0 manifest needed for target-identity
comparison in the fail-closed gate.

Zero file I/O, zero side effects.  Suitable for offline validation.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from federated_mcts.evaluation.game24_manifest import (
    _DECISION_FIELDS,
    _RUN_FIELDS,
    _compute_fingerprint,
    manifest_field_sets,
)


def _dict_to_canonical_subset(
    source: Mapping[str, object],
    field_set: frozenset[str],
) -> dict[str, object]:
    """Extract and normalize a canonical-key subset from a raw manifest dict.

    Mirrors ``Game24Manifest._to_canonical_dict`` but operates on a plain
    dict instead of a dataclass instance.  Includes every field present
    in *source* regardless of ``None`` values (exactly as the live
    dataclass does via ``hasattr``).  Raises ``ValueError`` (fail-closed)
    when a field in *field_set* is absent from *source*.

    Special handling:

    - ``remote_nondeterminism_notice`` is forced to ``True`` (constant)
    - ``schema_version`` is injected (from the module constant) when in the field set
    - List fields are sorted except ``task_ids`` (preserved order)
    """
    d: dict[str, object] = {}
    for name in sorted(field_set):
        if name == "remote_nondeterminism_notice":
            d[name] = True
        elif name == "schema_version":
            d[name] = source.get("schema_version", 1)
        else:
            if name not in source:
                raise ValueError(
                    f"B0 manifest missing required field {name!r} "
                    f"for canonical fingerprint reconstruction"
                )
            val = source[name]
            if isinstance(val, list):
                d[name] = sorted(val) if name != "task_ids" else list(val)
            else:
                d[name] = val
    return d


def derive_fingerprints_from_manifest_dict(
    manifest: Mapping[str, object],
) -> tuple[str, str]:
    """Derive (decision_policy_fingerprint, run_fingerprint) from manifest dict.

    Uses the same field sets and canonical-JSON hashing as
    ``Game24Manifest.decision_policy_fingerprint`` and
    ``Game24Manifest.run_fingerprint``.  This produces identical
    fingerprints to the live dataclass properties when the dict
    contains the same field values.

    Unknown keys are ignored; missing required fields in *field_set*
    raise ``ValueError`` (fail-closed).
    """
    version = manifest.get("schema_version", 1)
    if type(version) is not int:
        raise ValueError("Manifest schema_version must be an integer")
    decision_fields, run_fields = manifest_field_sets(version)
    if version == 2:
        from federated_mcts.config import DecisionSpec
        spec = DecisionSpec.validate_identity(manifest.get("decision_spec"))
        for name in set(spec) & _DECISION_FIELDS:
            expected = spec[name]
            if name == "seed" and expected is None:
                expected = 0
            if manifest.get(name) != expected:
                raise ValueError(f"Manifest projection disagrees with DecisionSpec: {name}")
    elif "decision_spec" in manifest:
        raise ValueError("Version-1 evidence cannot contain a decision_spec")
    decision_dict = _dict_to_canonical_subset(manifest, decision_fields)
    run_dict = _dict_to_canonical_subset(manifest, run_fields)
    dp_fp = _compute_fingerprint(decision_dict)
    run_fp = _compute_fingerprint(run_dict)
    return dp_fp, run_fp


# ═══════════════════════════════════════════════════════════════════
#  B0 source decision identity for target-identity gate
# ═══════════════════════════════════════════════════════════════════

@dataclass(frozen=True, slots=True)
class B0SourceDecisionIdentity:
    """Decision-affecting fields extracted from the validated B0 manifest.

    Used for fail-closed target-identity comparison: the target
    validation run's manifest MUST match these fields (except for
    task_id/split changes that legitimately differ between 20-pilot
    and 200-val).
    """

    model_config_sha256: str
    model_names: list[str]
    decision_policy: str
    seed: int
    temperature: float
    method_generate: str
    method_evaluate: str
    method_select: str
    n_generate_sample: int
    n_evaluate_sample: int
    n_select_sample: int
    game24_exact_prune: bool
    assign_strategy: str
    max_call_limit: int | None
    max_token_limit: int | None
    max_cost_usd: float | None
    max_raw_unit_limit: float
    runner_path: str
    schema_version: int = 1
    decision_spec: dict | None = None


def extract_source_decision_identity(
    manifest: Mapping[str, object],
) -> B0SourceDecisionIdentity:
    """Extract B0 decision identity fields from the validated manifest dict.

    All values are read from *manifest* without mutation.  Missing
    fields raise ``KeyError`` (fail-closed).  This function does NOT
    re-validate canonical expectations — the caller (``validate_b0_evidence``)
    has already performed field-level checks before invoking this.

    Field sources match ``Game24Manifest`` decision fields, consumed
    by the target-identity gate in ``enforce_b0_gate``.
    """
    def _s(key: str) -> str:
        v = manifest[key]
        if not isinstance(v, str):
            raise ValueError(f"B0 manifest {key!r} must be str, got {type(v).__name__}")
        return v

    def _i(key: str) -> int:
        v = manifest[key]
        if isinstance(v, bool):
            raise ValueError(f"B0 manifest {key!r} must be int, got bool")
        if not isinstance(v, (int, float)):
            raise ValueError(f"B0 manifest {key!r} must be int, got {type(v).__name__}")
        return int(v)

    def _f(key: str) -> float:
        v = manifest[key]
        if isinstance(v, bool):
            raise ValueError(f"B0 manifest {key!r} must be float, got bool")
        if not isinstance(v, (int, float)):
            raise ValueError(f"B0 manifest {key!r} must be float, got {type(v).__name__}")
        return float(v)

    def _b(key: str) -> bool:
        v = manifest[key]
        if not isinstance(v, bool):
            raise ValueError(f"B0 manifest {key!r} must be bool, got {type(v).__name__}")
        return v

    def _opt_i(key: str) -> int | None:
        v = manifest.get(key)
        if v is None:
            return None
        return _i(key)

    def _opt_f(key: str) -> float | None:
        v = manifest.get(key)
        if v is None:
            return None
        return _f(key)

    model_names_raw = manifest["model_names"]
    if not isinstance(model_names_raw, list):
        raise ValueError(
            f"B0 manifest model_names must be list, got {type(model_names_raw).__name__}"
        )
    model_names: list[str] = [str(n) for n in model_names_raw]
    version = manifest.get("schema_version", 1)
    if type(version) is not int:
        raise ValueError("Manifest schema_version must be an integer")
    manifest_field_sets(version)
    decision_spec = manifest.get("decision_spec")
    if decision_spec is not None and not isinstance(decision_spec, dict):
        raise ValueError("Manifest decision_spec must be an object")

    return B0SourceDecisionIdentity(
        model_config_sha256=_s("model_config_sha256"),
        model_names=model_names,
        decision_policy=_s("decision_policy"),
        seed=_i("seed"),
        temperature=_f("temperature"),
        method_generate=_s("method_generate"),
        method_evaluate=_s("method_evaluate"),
        method_select=_s("method_select"),
        n_generate_sample=_i("n_generate_sample"),
        n_evaluate_sample=_i("n_evaluate_sample"),
        n_select_sample=_i("n_select_sample"),
        game24_exact_prune=_b("game24_exact_prune"),
        assign_strategy=_s("assign_strategy"),
        max_call_limit=_opt_i("max_call_limit"),
        max_token_limit=_opt_i("max_token_limit"),
        max_cost_usd=_opt_f("max_cost_usd"),
        max_raw_unit_limit=_f("max_raw_unit_limit"),
        runner_path=_s("runner_path"),
        schema_version=version,
        decision_spec=decision_spec,
    )
