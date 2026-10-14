"""Metric definitions. Distributions are count vectors over the values 1..N.

Within-model collapse is computed per respondent; cross-model agreement is
computed over model respondents only (the PRNG control is scored separately).
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

METRIC_DEFINITIONS = {
    "normalized_entropy": "Miller-Madow corrected Shannon entropy (nats) of the parsed-answer distribution, "
    "H_MM = H_plugin + (m-1)/(2n) with m = occupied values and n = parsed answers, divided by ln(N). "
    "1 = uniform, 0 = point mass. 95% CI from a multinomial parametric bootstrap, re-centered on the "
    "point estimate (the bootstrap's own downward bias removed). Values slightly above 1 are possible for "
    "near-uniform sources because the correction can overshoot; they are reported, not clipped.",
    "mode": "Most frequent parsed value (ties broken toward the smaller value).",
    "p_mode": "Mode count / total responses (parse failures included in the denominator).",
    "top3_mass": "Mass on the three most frequent values / total responses.",
    "parse_failure_rate": "Responses that did not parse to exactly one in-range integer / total responses.",
    "pairwise_js": "Jensen-Shannon divergence, base 2 (0 identical, 1 disjoint), between parsed-answer "
    "distributions of every pair of models; full matrix plus mean over pairs.",
    "shared_mode_rate": "Fraction of model pairs with the same mode.",
    "top3_overlap": "Mean over model pairs of |top3(a) intersect top3(b)|, 0..3.",
    "single_guess_hit_rate_global": "v* = argmax_v mean_m p_m(v); hit = mean_m p_m(v*), p_m over all "
    "responses (failures are misses). Also reported leave-one-model-out: v* chosen from the other models, "
    "scored on the held-out one, which removes the in-sample advantage.",
    "single_guess_hit_rate_oracle": "mean_m max_v p_m(v): the attacker knows the model and guesses its mode.",
    "uniform_baseline": "1/N.",
}


def entropy_nats(counts: np.ndarray) -> float:
    n = counts.sum()
    if n == 0:
        return float("nan")
    p = counts[counts > 0] / n
    return float(-(p * np.log(p)).sum())


def miller_madow(counts: np.ndarray) -> float:
    n = counts.sum()
    m = int((counts > 0).sum())
    return entropy_nats(counts) + (m - 1) / (2 * n)


def normalized_entropy(counts: np.ndarray) -> float:
    return miller_madow(counts) / np.log(len(counts))


def entropy_ci(counts: np.ndarray, reps: int, rng: np.random.Generator, alpha=0.05) -> tuple[float, float]:
    n = int(counts.sum())
    if n == 0:
        return float("nan"), float("nan")
    boot = rng.multinomial(n, counts / n, size=reps)
    k = np.log(len(counts))
    occupied = (boot > 0).sum(1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p = boot / n
        h = -np.where(p > 0, p * np.log(p), 0).sum(1)
    vals = (h + (occupied - 1) / (2 * n)) / k
    # Resampling from the plug-in estimate is itself biased low, so re-center the
    # bootstrap spread on the point estimate (bias-corrected percentile interval).
    est, med = normalized_entropy(counts), float(np.median(vals))
    return est - (med - float(np.quantile(vals, alpha / 2))), est + (float(np.quantile(vals, 1 - alpha / 2)) - med)


def collapse_stats(counts: np.ndarray, n_total: int, reps: int, rng) -> dict:
    """counts over 1..N of parsed answers; n_total includes parse failures."""
    n_parsed = int(counts.sum())
    order = np.argsort(-counts, kind="stable")
    lo, hi = entropy_ci(counts, reps, rng) if n_parsed else (float("nan"), float("nan"))
    return {
        "n_total": n_total,
        "n_parsed": n_parsed,
        "parse_failure_rate": (n_total - n_parsed) / n_total if n_total else float("nan"),
        "normalized_entropy": normalized_entropy(counts) if n_parsed else float("nan"),
        "normalized_entropy_ci": [lo, hi],
        "mode": int(order[0] + 1) if n_parsed else None,
        "p_mode": float(counts[order[0]] / n_total) if n_parsed else 0.0,
        "top3": [int(v + 1) for v in order[:3]] if n_parsed else [],
        "top3_mass": float(counts[order[:3]].sum() / n_total) if n_parsed else 0.0,
    }


def js_divergence(p: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Base-2 JS divergence. p, q: (..., K) probability arrays; broadcasts over leading dims."""
    m = 0.5 * (p + q)
    with np.errstate(divide="ignore", invalid="ignore"):
        kp = np.where(p > 0, p * np.log2(p / m), 0.0).sum(-1)
        kq = np.where(q > 0, q * np.log2(q / m), 0.0).sum(-1)
    return 0.5 * (kp + kq)


def probs(counts: np.ndarray) -> np.ndarray:
    s = counts.sum(-1, keepdims=True)
    return np.divide(counts, s, out=np.zeros_like(counts, dtype=float), where=s > 0)


def mode_of(p: np.ndarray) -> int:
    return int(np.argmax(p))  # argmax takes the first max = smallest value


def top3_of(p: np.ndarray) -> set[int]:
    return set(np.argsort(-p, kind="stable")[:3].tolist())


def cross_model(parsed_counts: dict[str, np.ndarray], total: dict[str, int]) -> dict:
    """Agreement among models for one (range, temperature, prompt) or pooled group."""
    ids = sorted(parsed_counts)
    P = {i: probs(parsed_counts[i].astype(float)) for i in ids}
    pairs = list(combinations(ids, 2))
    js = {f"{a}~{b}": float(js_divergence(P[a], P[b])) for a, b in pairs}
    mean_js = float(np.mean(list(js.values()))) if pairs else float("nan")
    shared = float(np.mean([mode_of(P[a]) == mode_of(P[b]) for a, b in pairs])) if pairs else float("nan")
    overlap = float(np.mean([len(top3_of(P[a]) & top3_of(P[b])) for a, b in pairs])) if pairs else float("nan")
    hits = hit_rates(parsed_counts, total)
    return {"models": ids, "pairwise_js": js, "mean_pairwise_js": mean_js, "shared_mode_rate": shared,
            "top3_overlap": overlap, **hits}


def hit_rates(parsed_counts: dict[str, np.ndarray], total: dict[str, int]) -> dict:
    ids = sorted(parsed_counts)
    K = len(next(iter(parsed_counts.values())))
    # Mass over ALL responses: a parse failure is a miss for the attacker.
    A = np.stack([parsed_counts[i] / max(total[i], 1) for i in ids])
    avg = A.mean(0)
    v_star = int(np.argmax(avg))
    loo = []
    for j in range(len(ids)):
        others = np.delete(A, j, axis=0)
        if len(others) == 0:
            continue
        loo.append(A[j, int(np.argmax(others.mean(0)))])
    return {
        "global_guess": v_star + 1,
        "single_guess_hit_rate_global": float(avg[v_star]),
        "single_guess_hit_rate_global_loo": float(np.mean(loo)) if loo else float("nan"),
        "single_guess_hit_rate_oracle": float(A.max(1).mean()),
        "uniform_baseline": 1.0 / K,
    }
