"""Integrity checks run before anything is scored.

Hard failures (the run aborts, nothing is reported as a finding):
  * sample_counts: every cell holds exactly the predeclared number of responses
    (parsed or failed); nothing silently dropped, nothing extra.
  * control_uniform: in every (range, temperature>0) the PRNG control has zero
    parse failures and a normalized entropy no lower than the 0.1th percentile of
    what a true uniform source gives at the same sample size (simulated). A fixed
    threshold is wrong: Miller-Madow entropy of a genuinely uniform source with
    few samples per value sits well below 1.
  * control_outlier: at the headline temperatures, JS(control, model) exceeds the
    99.9th percentile of JS between two independent uniform samples of the same
    sizes, for EVERY model. That is: the pipeline can tell the known-uniform source
    apart from each model. (Comparing against mean model-model JS is not valid:
    models collapsed onto different values legitimately diverge from each other
    more than either does from uniform. That comparison is reported, not enforced.)

Soft (flagged and footnoted, not fatal):
  * determinism_t0: repeated temperature-0 calls return identical text.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import numpy as np

from harness.metrics import js_divergence, normalized_entropy
from harness.probe import build_cells, samples_for

SIM_REPS = 4000


class IntegrityError(RuntimeError):
    pass


def check_sample_counts(manifest, rows) -> dict:
    have = Counter((r["respondent"], r["n"], float(r["temperature"]), r["prompt_id"]) for r in rows)
    problems = []
    expected_keys = set()
    for c in build_cells(manifest):
        k = (c.respondent, c.n, c.temperature, c.prompt_id)
        expected_keys.add(k)
        want = samples_for(c, manifest)
        if have.get(k, 0) != want:
            problems.append(f"{c.key}: have {have.get(k, 0)}, want {want}")
    extra = [k for k in have if k not in expected_keys]
    problems += [f"unexpected cell {k}" for k in extra]
    return {"ok": not problems, "cells": len(expected_keys), "problems": problems[:50]}


def uniform_entropy_floor(n_samples: int, k: int, rng, q=0.001) -> float:
    sims = rng.multinomial(n_samples, np.full(k, 1 / k), size=SIM_REPS)
    return float(np.quantile([normalized_entropy(s) for s in sims], q))


def uniform_js_ceiling(n_a: int, n_b: int, k: int, rng, q=0.999) -> float:
    u = np.full(k, 1 / k)
    a = rng.multinomial(n_a, u, size=SIM_REPS) / max(n_a, 1)
    b = rng.multinomial(n_b, u, size=SIM_REPS) / max(n_b, 1)
    return float(np.quantile(js_divergence(a, b), q))


def check_control(manifest, pooled_stats, pooled_counts, control_js, pooled_cross, seed) -> dict:
    rng = np.random.default_rng(seed)
    cid = manifest["control_id"]
    out = {"uniform": [], "outlier": [], "ok": True}
    for (rid, n, t), s in pooled_stats.items():
        if rid != cid:
            continue
        floor = uniform_entropy_floor(s["n_parsed"], n, rng)
        ok = bool(s["normalized_entropy"] >= floor and s["parse_failure_rate"] == 0)
        out["uniform"].append({"n": n, "t": t, "normalized_entropy": s["normalized_entropy"],
                               "uniform_floor_p001": floor, "parse_failure_rate": s["parse_failure_rate"], "ok": ok})
        out["ok"] &= ok
    for (n, t), js_to_models in control_js.items():
        headline = t in manifest["headline_temperatures"]
        n_c = int(pooled_counts[(cid, n, t)].sum())
        per_model = {}
        for m, js in js_to_models.items():
            ceil = uniform_js_ceiling(n_c, int(pooled_counts[(m, n, t)].sum()), n, rng)
            per_model[m] = {"js": js, "uniform_vs_uniform_p999": ceil, "ok": bool(js > ceil)}
        ok = all(v["ok"] for v in per_model.values())
        out["outlier"].append({"n": n, "t": t, "per_model": per_model, "ok": ok, "enforced": headline,
                               "min_js_control_to_model": min(js_to_models.values()),
                               "mean_model_model_js_pooled": pooled_cross[(n, t)]["mean_pairwise_js"]})
        if headline:
            out["ok"] &= ok
    return out


def check_determinism(rows, control_id) -> dict:
    texts = defaultdict(set)
    for r in rows:
        if float(r["temperature"]) == 0.0 and r["respondent"] != control_id:
            texts[(r["respondent"], r["n"], r["prompt_id"])].add(r["text"].strip())
    nondet = sorted({k[0] for k, v in texts.items() if len(v) > 1})
    per_resp = defaultdict(lambda: [0, 0])
    for k, v in texts.items():
        per_resp[k[0]][0] += 1
        per_resp[k[0]][1] += len(v) > 1
    return {
        "nondeterministic_respondents": nondet,
        "cells_nondeterministic": {k: f"{v[1]}/{v[0]}" for k, v in per_resp.items()},
    }
