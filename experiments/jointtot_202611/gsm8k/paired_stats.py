"""Question-paired binary differences. No simulation or third-party packages."""

from collections import Counter
from fractions import Fraction
from math import comb


def empirical_sum_weights(plus, minus, zero):
    """Multinomial counts for all n^n ordered bootstrap samples of n pairs."""
    n = plus + minus + zero
    weights = Counter()
    for p in range(n + 1):
        for m in range(n + 1 - p):
            z = n - p - m
            w = comb(n, p) * comb(n - p, m) * plus**p * minus**m * zero**z
            if w:
                weights[p - m] += w
    return weights


def paired(tot, spec):
    if len(tot) != len(spec) or not tot or any(x not in (0, 1) for x in (*tot, *spec)):
        raise ValueError("need nonempty aligned binary outcomes")
    n = len(tot)
    plus = [i for i, (t, s) in enumerate(zip(tot, spec)) if s > t]
    minus = [i for i, (t, s) in enumerate(zip(tot, spec)) if s < t]
    np, nm = len(plus), len(minus)
    d = np + nm
    w = empirical_sum_weights(np, nm, n - d)
    total = sum(w.values())
    if total != n**n:
        raise ValueError("bootstrap count does not sum to n^n")
    def quantile(q):
        cumulative = 0
        for v in sorted(w):
            cumulative += w[v]
            if cumulative >= q * total:
                return float(Fraction(100 * v, n))
    exact_p = min(Fraction(1), Fraction(2 * sum(comb(d, k) for k in range(min(np, nm) + 1)), 2**d))
    both = sum(bool(t and s) for t, s in zip(tot, spec))
    return {"n_pairs": n, "tot_correct": sum(tot), "speculative_solve_correct": sum(spec),
            "both_correct": both, "neither_correct": n - both - d,
            "tot_only": nm, "speculative_solve_only": np,
            "tot_only_indices": minus, "speculative_solve_only_indices": plus,
            "spec_minus_tot_questions": np - nm, "spec_minus_tot_pp": 100 * (np - nm) / n,
            "paired_percentile_95_pp": [quantile(Fraction(1, 40)), quantile(Fraction(39, 40))],
            "exact_two_sided_mcnemar_p": float(exact_p)}
