"""The independent-bias null.

For every model, keep its realized distribution exactly (shape, entropy, degree
of collapse) but move that shape onto a random permutation of the value labels,
independently per model (and independently per prompt when several prompts are
aggregated). Recompute the statistic. This preserves how collapsed each model is
and destroys only the cross-model alignment, so the observed-vs-null gap measures
correlated collapse and nothing else.

Finite-sample JS bias is the same under the null as in the observation (the same
empirical shapes are used), so it cancels in the comparison.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np

from harness.metrics import js_divergence, probs


def _perm_stats(P: np.ndarray, A: np.ndarray, reps: int, rng: np.random.Generator):
    """P: (M, K) parsed-answer probabilities. A: (M, K) mass over all responses.

    Returns arrays (reps,) of mean pairwise JS, shared-mode rate, global hit rate
    under independent per-model relabeling.
    """
    M, K = P.shape
    idx = np.stack([rng.permuted(np.tile(np.arange(K), (reps, 1)), axis=1) for _ in range(M)])  # (M, reps, K)
    Pp = np.take_along_axis(np.broadcast_to(P[:, None, :], (M, reps, K)), idx, axis=2)
    Ap = np.take_along_axis(np.broadcast_to(A[:, None, :], (M, reps, K)), idx, axis=2)
    pairs = list(combinations(range(M), 2))
    js = np.mean([js_divergence(Pp[a], Pp[b]) for a, b in pairs], axis=0)
    modes = Pp.argmax(2)  # (M, reps)
    shared = np.mean([modes[a] == modes[b] for a, b in pairs], axis=0)
    hit = Ap.mean(0).max(1)
    return js, shared, hit


def independent_bias_null(groups: list[tuple[dict, dict]], reps: int, seed: int) -> dict:
    """groups: one (parsed_counts, total) per prompt, each over the same model set.

    The statistic is the mean over groups, matching how the observed value is aggregated.
    """
    rng = np.random.default_rng(seed)
    js_all, shared_all, hit_all = [], [], []
    obs_js, obs_shared, obs_hit = [], [], []
    for parsed, total in groups:
        ids = sorted(parsed)
        P = np.stack([probs(parsed[i].astype(float)) for i in ids])
        A = np.stack([parsed[i] / max(total[i], 1) for i in ids])
        js, sh, hit = _perm_stats(P, A, reps, rng)
        js_all.append(js); shared_all.append(sh); hit_all.append(hit)
        pairs = list(combinations(range(len(ids)), 2))
        obs_js.append(np.mean([js_divergence(P[a], P[b]) for a, b in pairs]))
        obs_shared.append(np.mean([P[a].argmax() == P[b].argmax() for a, b in pairs]))
        obs_hit.append(A.mean(0).max())
    null_js = np.mean(js_all, axis=0)
    null_shared = np.mean(shared_all, axis=0)
    null_hit = np.mean(hit_all, axis=0)
    o_js, o_sh, o_hit = float(np.mean(obs_js)), float(np.mean(obs_shared)), float(np.mean(obs_hit))
    return {
        "reps": reps,
        "observed_mean_js": o_js,
        "null_mean_js": float(null_js.mean()),
        "null_js_p01": float(np.quantile(null_js, 0.01)),
        "null_js_p50": float(np.quantile(null_js, 0.5)),
        "null_js_min": float(null_js.min()),
        # one-sided: how often independent relabeling is at least as aligned as reality
        "p_value_js": float((np.sum(null_js <= o_js) + 1) / (reps + 1)),
        "js_ratio_observed_over_null": o_js / float(null_js.mean()) if null_js.mean() > 0 else float("nan"),
        "observed_shared_mode_rate": o_sh,
        "null_shared_mode_rate": float(null_shared.mean()),
        "p_value_shared_mode": float((np.sum(null_shared >= o_sh) + 1) / (reps + 1)),
        "observed_global_hit": o_hit,
        "null_global_hit": float(null_hit.mean()),
        "p_value_global_hit": float((np.sum(null_hit >= o_hit) + 1) / (reps + 1)),
    }
