import numpy as np
import pytest

from harness.metrics import collapse_stats, cross_model, js_divergence, normalized_entropy
from harness.null_model import independent_bias_null
from harness.parse import parse_answer, parse_answer_lenient


@pytest.mark.parametrize("text,n,want", [
    ("7", 10, (7, "ok")),
    ("**7**", 10, (7, "ok")),
    ("The random number is... 7!", 10, (7, "ok")),
    ("Sure! A random number between 1 and 10: 3.", 10, (3, "ok")),
    ("I'll pick a number from 1 to 100: 42. Your number is 42!", 100, (42, "ok")),
    ("Here's one from 1-100: 73", 100, (73, "ok")),
    ("...5!", 10, (5, "ok")),
    ("seven", 10, (7, "ok")),
    ("<think>maybe 3</think>8", 10, (8, "ok")),
    ("3 or 8", 10, (None, "multiple_numbers")),
    ("How about **17**? It's prime, and 1 + 7 = 8.", 20, (17, "ok")),
    ("**3** or **8**", 10, (None, "multiple_numbers")),
    ("I can't, but:\n\n1. **Use a generator**\n2. Roll a die", 10, (None, "no_number")),
    ("Methods:\n1. Think of 4\n2. Done", 10, (4, "ok")),
    ("0", 10, (None, "out_of_range")),
    ("137", 100, (None, "out_of_range")),
    ("7.5", 10, (None, "non_integer")),
    ("I can't generate true randomness.", 10, (None, "no_number")),
    ("Between 1 and 10? Okay.", 10, (None, "no_number")),
    ("10", 10, (10, "ok")),
    ("1", 10, (1, "ok")),
])
def test_parse(text, n, want):
    assert parse_answer(text, n) == want


def test_uniform_entropy_near_one():
    rng = np.random.default_rng(0)
    for k in (10, 100):
        c = np.bincount(rng.integers(0, k, 1000), minlength=k)
        assert normalized_entropy(c) > 0.97
        s = collapse_stats(c, 1000, 200, rng)
        lo, hi = s["normalized_entropy_ci"]
        assert lo < s["normalized_entropy"] < hi + 1e-9


def test_point_mass_entropy_zero():
    c = np.zeros(10, int); c[6] = 500
    assert normalized_entropy(c) == pytest.approx(0.0)


def test_js_bounds():
    p = np.array([1.0, 0, 0]); q = np.array([0, 1.0, 0])
    assert js_divergence(p, q) == pytest.approx(1.0)
    assert js_divergence(p, p) == pytest.approx(0.0)


def _biased(rng, k, mode, n=400, peak=0.5):
    p = np.full(k, (1 - peak) / (k - 1)); p[mode] = peak
    return rng.multinomial(n, p)


def test_null_detects_shared_bias():
    rng = np.random.default_rng(1)
    groups = [({f"m{i}": _biased(rng, 10, 6) for i in range(5)}, {f"m{i}": 400 for i in range(5)}) for _ in range(3)]
    r = independent_bias_null(groups, 500, seed=2)
    assert r["p_value_js"] < 0.01
    assert r["observed_mean_js"] < r["null_js_p01"]


def test_null_does_not_fire_on_independent_bias():
    """Equally collapsed models, each onto a different value: biased but not correlated."""
    rng = np.random.default_rng(3)
    groups = [({f"m{i}": _biased(rng, 10, i) for i in range(5)}, {f"m{i}": 400 for i in range(5)}) for _ in range(3)]
    r = independent_bias_null(groups, 500, seed=4)
    assert r["p_value_js"] > 0.5


def test_hit_rates():
    c = {"a": np.array([0, 0, 0, 0, 0, 0, 80, 20, 0, 0]), "b": np.array([0, 0, 0, 0, 0, 0, 50, 0, 50, 0])}
    r = cross_model(c, {"a": 100, "b": 100})
    assert r["global_guess"] == 7
    assert r["single_guess_hit_rate_global"] == pytest.approx(0.65)
    assert r["single_guess_hit_rate_oracle"] == pytest.approx(0.65)
    assert r["shared_mode_rate"] == 1.0


def test_lenient():
    assert parse_answer_lenient("I pick 7, since there are 9 others", 10) == (7, "ok")
    assert parse_answer_lenient("between 1 and 10: 0 then 4", 10) == (4, "ok")
    assert parse_answer_lenient("I can't.\n\n1. **Use a generator**\n2. Search online", 10) == (None, "no_number")
