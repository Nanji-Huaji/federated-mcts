"""Exact dead-end pruning for Game24 using Fraction-based exhaustive search.

Provides a side-effect-free oracle ``prune_unreachable`` that removes
candidates whose remaining-number multiset cannot reach exactly 24 through
any legal sequence of +, -, *, / operations.  The oracle is memoized,
uses fractions.Fraction for exact arithmetic, and never exposes oracle
labels to model selection.

Requirements (from the task specification):
  - fractions.Fraction arithmetic only (no float)
  - Both operand orders for subtraction and division
  - Division by zero is skipped
  - Memoized over the sorted Fraction tuple
  - Returns (retained_list, proposed_count, pruned_count)
"""

from __future__ import annotations

from fractions import Fraction
from functools import lru_cache

from federated_mcts.tasks.game24 import get_current_numbers


@lru_cache(maxsize=None)
def _can_reach_target_numbers(
    numbers: tuple[Fraction, ...],
    target: Fraction = Fraction(24),
) -> bool:
    """Return True iff the *sorted* multiset of Fractions can be combined
    via +, -, *, / to yield exactly *target*.

    The search enumerates every unordered pair (a, b) from *numbers*,
    evaluates all six legal operations (including both operand orders for
    non-commutative subtraction and division), and recurses on the
    reduced multiset.  Memoization via ``functools.lru_cache`` ensures
    identical multisets are computed only once.

    Parameters
    ----------
    numbers:
        Tuple of Fractions **sorted in ascending order**.  The caller
        must guarantee this invariant; the cache key depends on it.
    target:
        The target Fraction (default: 24).
    """
    n = len(numbers)
    if n == 1:
        return numbers[0] == target

    for i in range(n):
        a = numbers[i]
        for j in range(i + 1, n):
            b = numbers[j]

            # Build the "rest" list (n-2 remaining numbers)
            rest: list[Fraction] = []
            for k in range(n):
                if k != i and k != j:
                    rest.append(numbers[k])

            # --- a + b (commutative) ---
            if _can_reach_target_numbers(
                tuple(sorted(rest + [a + b])), target,
            ):
                return True

            # --- a * b (commutative) ---
            if _can_reach_target_numbers(
                tuple(sorted(rest + [a * b])), target,
            ):
                return True

            # --- a - b, b - a (both orders) ---
            if _can_reach_target_numbers(
                tuple(sorted(rest + [a - b])), target,
            ):
                return True

            if _can_reach_target_numbers(
                tuple(sorted(rest + [b - a])), target,
            ):
                return True

            # --- a / b (if divisor non-zero) ---
            if b != 0:
                if _can_reach_target_numbers(
                    tuple(sorted(rest + [Fraction(a, b)])), target,
                ):
                    return True

            # --- b / a (if divisor non-zero) ---
            if a != 0:
                if _can_reach_target_numbers(
                    tuple(sorted(rest + [Fraction(b, a)])), target,
                ):
                    return True

    return False


def prune_unreachable(
    candidates: list[str],
    task,
    x: str,
) -> tuple[list[str], int, int]:
    """Remove dead-end candidates whose remaining numbers cannot reach 24.

    Parameters
    ----------
    candidates:
        Proposed candidates (e.g. ``"3 + 4 = 7 (left: 5 7)"``).
    task:
        A ``Game24Task`` instance (provides ``get_current_numbers`` /
        ``canonical_state_key`` via ``get_current_numbers`` helper).
    x:
        The original problem input string (e.g. ``"1 2 3 4"``).

    Returns
    -------
    (retained, proposed, pruned):
        *retained* contains only the reachable candidates in their
        original relative order.  *proposed* and *pruned* are the
        aggregate counts.
    """
    proposed = len(candidates)
    if proposed == 0:
        return [], 0, 0

    retained: list[str] = []
    pruned = 0

    for candidate in candidates:
        current_str = get_current_numbers(candidate if candidate.strip() else x)
        nums = tuple(
            sorted(Fraction(num) for num in current_str.split())
        )

        # A terminal [24] is trivially reachable.
        if len(nums) == 1 and nums[0] == Fraction(24):
            retained.append(candidate)
        elif _can_reach_target_numbers(nums):
            retained.append(candidate)
        else:
            pruned += 1

    return retained, proposed, pruned
