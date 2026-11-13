"""Fail-closed comparison of a validation run with measured pilot evidence."""
from __future__ import annotations

from dataclasses import fields
import sys

from federated_mcts.config import DecisionSpec
from federated_mcts.evaluation.game24_b0_evidence import B0EvidenceRecord
from federated_mcts.evaluation.game24_manifest import Game24Manifest

_ALLOWED_DIFF_FIELDS = frozenset({
    "task_ids", "ordered_task_ids_sha256", "task_split_name", "split_file_sha256",
})


def enforce_target_identity(target: Game24Manifest, evidence: B0EvidenceRecord) -> None:
    """Compare the entire declared identity, not another hand-copied field list.

    Legacy evidence remains verifiable, but cannot authorize a version-2 run:
    omitted decision options cannot safely be inferred from a historical hash.
    """
    source = evidence.source_decision
    mismatches: list[str] = []
    if target.task != "game24":
        mismatches.append(f"task: expected 'game24', got {target.task!r}")
    if len(target.task_ids) != 200:
        mismatches.append(f"task_ids count: expected 200, got {len(target.task_ids)}")
    if target.task_split_name != "val":
        mismatches.append("task_split_name: full validation requires 'val'")

    for item in fields(source):
        name = item.name
        expected, actual = getattr(source, name), getattr(target, name)
        if name == "model_names":
            expected, actual = sorted(expected), sorted(actual)
        if isinstance(actual, DecisionSpec):
            actual = actual.identity()
        if actual != expected:
            if name == "decision_spec" and isinstance(expected, dict) and isinstance(actual, dict):
                mismatches.extend(
                    f"decision_spec.{key}: expected {expected.get(key)!r}, got {actual.get(key)!r}"
                    for key in sorted(expected.keys() | actual.keys())
                    if expected.get(key) != actual.get(key))
            else:
                mismatches.append(f"{name}: expected {expected!r}, got {actual!r}")
    if mismatches:
        print("FATAL: Target-identity gate FAILED — target decision fields do not match B0 source evidence.", file=sys.stderr)
        for message in mismatches:
            print(f"  {message}", file=sys.stderr)
        if target.schema_version != source.schema_version:
            print("  Legacy evidence is read-only; collect a new version-2 pilot before a new validation run.", file=sys.stderr)
        print(f"Allowed differences: {', '.join(sorted(_ALLOWED_DIFF_FIELDS))}.", file=sys.stderr)
        raise SystemExit(1)
    print("Target-identity gate PASSED — decision fields match B0 source evidence. "
          f"(model_config_sha256={source.model_config_sha256[:12]}..., "
          f"model_names={source.model_names}, decision_fp={evidence.decision_policy_fingerprint[:12]}...)")
