"""Exact tests used in the paper: two-sided McNemar and binomial."""
from __future__ import annotations

import math


def log_comb(n: int, k: int) -> float:
    if k < 0 or k > n:
        return float("-inf")
    return math.lgamma(n + 1) - math.lgamma(k + 1) - math.lgamma(n - k + 1)


def exact_mcnemar_p(b: int, c: int) -> float:
    """Exact two-sided McNemar on discordant counts (b, c), p = 0.5."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return exact_binomial_two_sided(k, n)


def exact_binomial_two_sided(k: int, n: int, p: float = 0.5) -> float:
    """P(X <= k or X >= n-k) for X ~ Bin(n, p), with k = min(count, n-count)."""
    if n == 0:
        return 1.0
    k = min(k, n - k)
    logp = math.log(p)
    logq = math.log(1.0 - p)
    terms = [log_comb(n, i) + i * logp + (n - i) * logq for i in range(k + 1)]
    m = max(terms)
    cdf = math.exp(m) * sum(math.exp(t - m) for t in terms)
    return min(1.0, 2.0 * cdf)
