"""Paired Game24 experiment report formatter.

Generates matching JSON and Markdown output from a ``PairedAnalysis``.
Kept separate from ``game24_analysis.py`` to stay under the 250-LOC cap.
"""

from __future__ import annotations

import json
from typing import Any

from federated_mcts.evaluation.game24_analysis import PairedAnalysis


def format_report(analysis: PairedAnalysis) -> tuple[str, str]:
    """Generate (json_str, markdown_str) from a paired analysis.

    Both outputs agree exactly on all numbers.  JSON is the
    machine-readable source of truth; Markdown is for human review.
    """
    base = analysis.baseline
    cand = analysis.candidate

    # ── JSON payload ──────────────────────────────────────────────
    j: dict[str, Any] = {
        "schema": "game24_paired_analysis/v1",
        "baseline_accuracy": analysis.baseline_accuracy,
        "candidate_accuracy": analysis.candidate_accuracy,
        "delta": round(analysis.delta, 10),
        "ci_95_lower": analysis.ci_95_lower,
        "ci_95_upper": analysis.ci_95_upper,
        "ci_95_method": (
            "paired_bootstrap_n=10000_seed=%d_alpha=0.05_descriptive_only"
            % analysis.bootstrap_seed
        ),
        "baseline": {
            "decision_policy": base.decision_policy,
            "decision_policy_fingerprint": base.decision_policy_fingerprint,
            "run_fingerprint": base.run_fingerprint,
            "task_count": base.total_tasks,
            "solved_count": base.solved_count,
            "total_calls": base.total_calls,
            "total_tokens": base.total_tokens,
            "estimated_cost_usd": base.cost,
            "complete": base.complete,
        },
        "candidate": {
            "decision_policy": cand.decision_policy,
            "decision_policy_fingerprint": cand.decision_policy_fingerprint,
            "run_fingerprint": cand.run_fingerprint,
            "task_count": cand.total_tasks,
            "solved_count": cand.solved_count,
            "total_calls": cand.total_calls,
            "total_tokens": cand.total_tokens,
            "estimated_cost_usd": cand.cost,
            "complete": cand.complete,
        },
        "pareto_rows": {
            "baseline": {
                "accuracy": analysis.baseline_accuracy,
                "cost": base.cost,
                "calls": base.total_calls,
                "tokens": base.total_tokens,
            },
            "candidate": {
                "accuracy": analysis.candidate_accuracy,
                "cost": cand.cost,
                "calls": cand.total_calls,
                "tokens": cand.total_tokens,
            },
        },
        "validation_errors": analysis.validation_errors,
    }

    json_str = json.dumps(j, indent=2, sort_keys=True, allow_nan=False)

    # ── Markdown report ───────────────────────────────────────────
    lines: list[str] = _build_markdown(analysis, base, cand)
    md_str = "\n".join(lines) + "\n"

    return json_str, md_str


def _build_markdown(
    analysis: PairedAnalysis,
    base: Any,
    cand: Any,
) -> list[str]:
    """Build Markdown report lines."""
    out: list[str] = []
    out.append("# Paired Game24 Analysis Report")
    out.append("")
    out.append("## Summary")
    out.append("")
    out.append("| Metric | Baseline | Candidate | Delta |")
    out.append("|---|---|---|---|")
    out.append(
        f"| Solved / Total | {base.solved_count}/{base.total_tasks} | "
        f"{cand.solved_count}/{cand.total_tasks} | — |"
    )
    out.append(
        f"| Accuracy | {analysis.baseline_accuracy:.4f} | "
        f"{analysis.candidate_accuracy:.4f} | "
        f"{analysis.delta:+.4f} |"
    )
    out.append(
        f"| 95% CI (descriptive) | — | — | "
        f"[{analysis.ci_95_lower:+.4f}, {analysis.ci_95_upper:+.4f}] |"
    )
    out.append("")
    out.append("## Resources")
    out.append("")
    out.append("| Resource | Baseline | Candidate |")
    out.append("|---|---|")
    out.append(f"| Total Calls | {base.total_calls} | {cand.total_calls} |")
    out.append(f"| Total Tokens | {base.total_tokens} | {cand.total_tokens} |")
    out.append(f"| Estimated Cost (USD) | {base.cost:.6f} | {cand.cost:.6f} |")
    out.append("")
    out.append("## Identity")
    out.append("")
    out.append(f"- **Baseline policy**: `{base.decision_policy}`")
    out.append(f"- **Candidate policy**: `{cand.decision_policy}`")
    out.append(f"- **Baseline decision fingerprint**: `{base.decision_policy_fingerprint}`")
    out.append(f"- **Candidate decision fingerprint**: `{cand.decision_policy_fingerprint}`")
    out.append(f"- **Baseline run fingerprint**: `{base.run_fingerprint}`")
    out.append(f"- **Candidate run fingerprint**: `{cand.run_fingerprint}`")
    out.append("")
    if analysis.validation_errors:
        out.append("## Validation Errors")
        out.append("")
        for err in analysis.validation_errors:
            out.append(f"- {err}")
        out.append("")
    out.append("---")
    out.append(
        "*Intervals are descriptive (not confirmatory).  "
        "Do not interpret bootstrap CIs as inferential evidence of superiority.*"
    )
    return out
