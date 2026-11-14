#!/usr/bin/env python3
"""Paired Game24 experiment analysis and report CLI (Todo 4).

Subcommands:
  analyze    Run paired analysis and produce JSON + Markdown
  verify     Validate manifests/results for integrity
  lint-report Check a report for correctness and completeness
"""

from __future__ import annotations

import argparse
import json
import os
import sys

# Ensure the src directory is on sys.path when run from repo root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))


def _arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="analyze_game24",
        description="Paired Game24 experiment analysis and report CLI",
    )
    sub = p.add_subparsers(dest="command", required=True)

    # ── analyze ───────────────────────────────────────────────────
    ana = sub.add_parser("analyze", help="Run paired analysis")
    ana.add_argument("--baseline-manifest", required=True)
    ana.add_argument("--baseline-performance", required=True)
    ana.add_argument("--baseline-tasks", required=True)
    ana.add_argument("--candidate-manifest", required=True)
    ana.add_argument("--candidate-performance", required=True)
    ana.add_argument("--candidate-tasks", required=True)
    ana.add_argument("--output-json", default=None)
    ana.add_argument("--output-md", default=None)
    ana.add_argument("--seed", type=int, default=42,
                     help="Bootstrap seed (default: 42)")
    ana.add_argument("--bootstrap-n", type=int, default=10_000,
                     help="Bootstrap resamples (default: 10000)")

    # ── verify ────────────────────────────────────────────────────
    ver = sub.add_parser("verify", help="Validate manifests/results")
    ver.add_argument("--manifest", required=True)
    ver.add_argument("--performance", required=True)
    ver.add_argument("--tasks", required=True)
    ver.add_argument("--evidence-root", default=None,
                     help="Root for evidence output")

    # ── lint-report ───────────────────────────────────────────────
    lnt = sub.add_parser("lint-report", help="Check report correctness")
    lnt.add_argument("--report", required=True,
                     help="Path to Markdown report to lint")
    lnt.add_argument("--results", required=True,
                     help="Path to JSON results for cross-check")
    return p


# ═══════════════════════════════════════════════════════════════════
#  Commands
# ═══════════════════════════════════════════════════════════════════

def cmd_analyze(args: argparse.Namespace) -> int:
    """Run paired analysis and produce JSON + Markdown output."""
    from federated_mcts.evaluation.game24_analysis import (
        load_result,
        run_paired_analysis,
    )
    from federated_mcts.evaluation.game24_report import format_report

    try:
        baseline = load_result(
            args.baseline_manifest,
            args.baseline_performance,
            args.baseline_tasks,
        )
        candidate = load_result(
            args.candidate_manifest,
            args.candidate_performance,
            args.candidate_tasks,
        )
    except (FileNotFoundError, ValueError) as exc:
        print(f"Error loading results: {exc}")
        return 1

    try:
        analysis = run_paired_analysis(
            baseline, candidate, seed=args.seed
        )
    except ValueError as exc:
        print(f"Error in paired analysis: {exc}")
        return 2

    json_str, md_str = format_report(analysis)

    # Write outputs
    if args.output_json:
        with open(args.output_json, "w") as f:
            f.write(json_str)
        print(f"JSON report → {args.output_json}")
    if args.output_md:
        with open(args.output_md, "w") as f:
            f.write(md_str)
        print(f"Markdown report → {args.output_md}")

    # Always print summary to stdout
    if not args.output_json and not args.output_md:
        print(json_str)
        print("\n---\n")
        print(md_str)

    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    """Validate experiment manifests and results for integrity."""
    from federated_mcts.evaluation.game24_analysis import load_result

    errors: list[str] = []
    try:
        result = load_result(
            args.manifest, args.performance, args.tasks
        )
    except FileNotFoundError as exc:
        errors.append(f"Missing file: {exc}")
    except ValueError as exc:
        errors.append(f"Validation failed: {exc}")
    else:
        print("\u2713 Manifest loaded")
        print(f"  Task IDs: {result.task_ids[:5]}{'...' if len(result.task_ids) > 5 else ''}")
        print(f"  Total tasks: {result.total_tasks}")
        print(f"  Solved: {result.solved_count}/{result.total_tasks}")
        print(f"  Accuracy: {result.accuracy:.4f}")
        print(f"  Decision policy: {result.decision_policy}")
        print(f"  Decision fingerprint: {result.decision_policy_fingerprint}")
        print(f"  Run fingerprint: {result.run_fingerprint}")
        print(f"  Complete: {result.complete}")
        print(f"  Cost: {result.cost:.6f}")
        print(f"  Calls: {result.total_calls}")
        print(f"  Tokens: {result.total_tokens}")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        return 1
    return 0


def cmd_lint_report(args: argparse.Namespace) -> int:
    """Lint a Markdown report against the source-of-truth JSON."""
    errors: list[str] = []

    if not os.path.exists(args.report):
        print(f"✗ Report file not found: {args.report}")
        return 1
    if not os.path.exists(args.results):
        print(f"✗ Results file not found: {args.results}")
        return 1

    with open(args.report) as f:
        md_content = f.read()
    with open(args.results) as f:
        try:
            js = json.load(f)
        except json.JSONDecodeError as exc:
            print(f"✗ Results JSON is invalid: {exc}")
            return 1

    # Required fields in report
    required_terms = ["solved", "total", "delta", "ci", "policy", "fingerprint"]
    md_lower = md_content.lower()
    for term in required_terms:
        if term not in md_lower:
            errors.append(f"Markdown missing '{term}'")

    # Cross-check JSON → MD agreement
    json_accuracy = js.get("candidate_accuracy")
    if json_accuracy is not None:
        acc_str = f"{json_accuracy:.4f}"
        if acc_str not in md_content:
            errors.append(
                f"Candidate accuracy {acc_str} not found in Markdown"
            )

    baseline_acc = js.get("baseline_accuracy")
    cand_acc = js.get("candidate_accuracy")
    if baseline_acc is not None and cand_acc is not None:
        delta = cand_acc - baseline_acc
        delta_str = f"{delta:+.4f}"
        if delta_str not in md_content:
            errors.append(f"Delta {delta_str} not found in Markdown")

    if errors:
        for e in errors:
            print(f"✗ {e}")
        return 1

    print("✓ Report lint passes")
    print(f"  Baseline accuracy: {baseline_acc}")
    print(f"  Candidate accuracy: {cand_acc}")
    print(f"  Delta: {delta_str}")
    return 0


# ═══════════════════════════════════════════════════════════════════
#  Entry point
# ═══════════════════════════════════════════════════════════════════

def main() -> int:
    parser = _arg_parser()
    args = parser.parse_args()

    if args.command == "analyze":
        return cmd_analyze(args)
    elif args.command == "verify":
        return cmd_verify(args)
    elif args.command == "lint-report":
        return cmd_lint_report(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
