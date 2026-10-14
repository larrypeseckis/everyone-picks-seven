"""Run, score, report.

    python -m harness.controller fingerprint
    python -m harness.controller freeze             # pin fingerprints, hash manifest + scoring code
    python -m harness.controller sample [--pilot]   # canonical sampling refuses without a valid freeze
    python -m harness.controller score  [--pilot]
    python -m harness.controller report [--pilot]
    python -m harness.controller run    [--pilot]   # sample + score + report
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from harness import integrity
from harness.metrics import METRIC_DEFINITIONS, collapse_stats, cross_model, js_divergence, probs
from harness.null_model import independent_bias_null
from harness.parse import PARSE_RULE, parse_answer, parse_answer_lenient
from harness.sample import load_responses, run_sampling
from respondents.base import make_respondent

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "manifest.json"
LOCK = ROOT / "manifest.lock"
# The scoring definition: if any of these change after the freeze, scoring refuses.
FROZEN_CODE = ["harness/parse.py", "harness/metrics.py", "harness/null_model.py", "harness/integrity.py",
               "harness/probe.py", "harness/controller.py"]


def sha256(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def load_manifest(pilot: bool) -> dict:
    m = json.loads(MANIFEST.read_text())
    if pilot:
        m = copy.deepcopy(m)
        m.update(m["pilot"]["overrides"])
    return m


def evidence_dir(pilot: bool) -> Path:
    return ROOT / ("evidence/pilot" if pilot else "evidence")


# ---------------------------------------------------------------- freeze

def current_hashes() -> dict:
    return {"manifest.json": sha256(MANIFEST.read_bytes()),
            **{p: sha256((ROOT / p).read_bytes()) for p in FROZEN_CODE}}


def live_fingerprints(manifest) -> dict:
    return {r["id"]: make_respondent(r).fingerprint() for r in manifest["respondents"]}


def cmd_freeze(args):
    if LOCK.exists() and not args.force:
        sys.exit("manifest.lock exists. Refusing to refreeze (use --force and record why in AMENDMENTS.md).")
    m = json.loads(MANIFEST.read_text())
    m["fingerprints"] = live_fingerprints(m)
    m["parse_rule"] = PARSE_RULE
    m["metric_definitions"] = METRIC_DEFINITIONS
    m["frozen_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    MANIFEST.write_text(json.dumps(m, indent=2) + "\n")
    lock = {"frozen_at": m["frozen_at"], "sha256": current_hashes()}
    LOCK.write_text(json.dumps(lock, indent=2) + "\n")
    print(f"frozen. manifest sha256 {lock['sha256']['manifest.json']}")


def verify_lock(check_live=False):
    if not LOCK.exists():
        sys.exit("no manifest.lock: run `freeze` before the canonical run (or use --pilot).")
    lock = json.loads(LOCK.read_text())
    now = current_hashes()
    bad = [k for k, v in lock["sha256"].items() if now.get(k) != v]
    if bad:
        sys.exit(f"REFUSING: frozen files changed since freeze: {bad}")
    if check_live:
        m = json.loads(MANIFEST.read_text())
        live = live_fingerprints(m)
        drift = [rid for rid, fp in m["fingerprints"].items() if live.get(rid) != fp]
        if drift:
            sys.exit(f"REFUSING: respondent fingerprints drifted since freeze: {drift}")
    return lock


# ---------------------------------------------------------------- score

def _counts(rows, n):
    c = np.zeros(n, dtype=np.int64)
    for r in rows:
        if r["status"] == "ok":
            c[r["value"] - 1] += 1
    return c


def score(manifest, rows) -> dict:
    rng = np.random.default_rng(manifest["seed"])
    reps_boot = manifest["bootstrap_reps"]
    control = manifest["control_id"]
    models = [r["id"] for r in manifest["respondents"] if not r.get("control")]
    temps_hot = [t for t in manifest["temperatures"] if t > 0]

    groups = defaultdict(list)
    for r in rows:
        groups[(r["respondent"], r["n"], float(r["temperature"]), r["prompt_id"])].append(r)
    prompt_ids = [p["id"] for p in manifest["prompts"]]

    # -- integrity: counts first, before computing anything that could be reported
    counts_check = integrity.check_sample_counts(manifest, rows)
    if not counts_check["ok"]:
        raise integrity.IntegrityError(f"sample counts: {counts_check['problems'][:10]}")

    # -- within-model collapse, per cell and pooled over prompts
    per_cell, pooled, pooled_counts, pooled_total = [], {}, {}, {}
    failure_reasons = defaultdict(lambda: defaultdict(int))
    for rid in [r["id"] for r in manifest["respondents"]]:
        for n in manifest["ranges"]:
            for t in temps_hot:
                cell_rows = []
                for pid in prompt_ids:
                    g = groups.get((rid, n, t, pid), [])
                    if not g:
                        continue
                    cell_rows += g
                    s = collapse_stats(_counts(g, n), len(g), reps_boot, rng)
                    per_cell.append({"respondent": rid, "n": n, "t": t, "prompt_id": pid, **s})
                    for r in g:
                        if r["status"] != "ok":
                            failure_reasons[rid][r["status"]] += 1
                if cell_rows:
                    c = _counts(cell_rows, n)
                    pooled[(rid, n, t)] = collapse_stats(c, len(cell_rows), reps_boot, rng)
                    pooled_counts[(rid, n, t)] = c
                    pooled_total[(rid, n, t)] = len(cell_rows)

    # -- cross-model agreement (models only), per prompt, aggregated, and pooled; plus the null
    cross, null = {}, {}
    for n in manifest["ranges"]:
        for t in temps_hot:
            present = [m for m in models if (m, n, t) in pooled_counts]
            if len(present) < 2:
                continue
            per_prompt, null_groups = [], []
            for pid in prompt_ids:
                pc = {m: _counts(groups[(m, n, t, pid)], n) for m in present}
                tot = {m: len(groups[(m, n, t, pid)]) for m in present}
                per_prompt.append({"prompt_id": pid, **cross_model(pc, tot)})
                null_groups.append((pc, tot))
            pooled_cross = cross_model({m: pooled_counts[(m, n, t)] for m in present},
                                       {m: pooled_total[(m, n, t)] for m in present})
            agg = {}
            for k in ["mean_pairwise_js", "shared_mode_rate", "top3_overlap", "single_guess_hit_rate_global",
                      "single_guess_hit_rate_global_loo", "single_guess_hit_rate_oracle"]:
                vals = [p[k] for p in per_prompt]
                agg[k] = {"mean": float(np.mean(vals)), "sd": float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
                          "min": float(np.min(vals)), "max": float(np.max(vals))}
            cross[(n, t)] = {"per_prompt": per_prompt, "aggregate_over_prompts": agg, "pooled": pooled_cross,
                             "mean_pairwise_js": agg["mean_pairwise_js"]["mean"]}
            null[(n, t)] = independent_bias_null(null_groups, manifest["null_reps"],
                                                 seed=manifest["seed"] + n * 1000 + int(t * 10))

    # -- control vs models
    control_js = {}
    for n in manifest["ranges"]:
        for t in temps_hot:
            if (control, n, t) not in pooled_counts or (n, t) not in cross:
                continue
            pc = probs(pooled_counts[(control, n, t)].astype(float))
            control_js[(n, t)] = {m: float(js_divergence(pc, probs(pooled_counts[(m, n, t)].astype(float))))
                                  for m in models if (m, n, t) in pooled_counts}

    control_check = integrity.check_control(manifest, pooled, pooled_counts, control_js,
                                            {k: v["pooled"] for k, v in cross.items()}, manifest["seed"] + 7)
    determinism = integrity.check_determinism(rows, control)
    if not control_check["ok"]:
        raise integrity.IntegrityError(f"uniform control failed: {json.dumps(control_check, indent=1)}")

    # -- greedy (temperature 0): the mode and whether it is shared
    greedy = []
    for n in manifest["ranges"]:
        for pid in prompt_ids:
            modes = {}
            for m in models:
                g = groups.get((m, n, 0.0, pid), [])
                vals = [r["value"] for r in g if r["status"] == "ok"]
                modes[m] = max(set(vals), key=vals.count) if vals else None
            got = [v for v in modes.values() if v is not None]
            greedy.append({"n": n, "prompt_id": pid, "modes": modes,
                           "most_common": max(set(got), key=got.count) if got else None,
                           "share_most_common": got.count(max(set(got), key=got.count)) / len(models) if got else 0})

    results = {
        "integrity": {"sample_counts": counts_check, "control": control_check, "determinism_t0": determinism},
        "per_cell": per_cell,
        "pooled": [{"respondent": k[0], "n": k[1], "t": k[2], **v} for k, v in pooled.items()],
        "pooled_counts": [{"respondent": k[0], "n": k[1], "t": k[2], "counts": v.tolist()}
                          for k, v in pooled_counts.items()],
        "failure_reasons": {k: dict(v) for k, v in failure_reasons.items()},
        "cross": [{"n": k[0], "t": k[1], **v} for k, v in cross.items()],
        "null": [{"n": k[0], "t": k[1], **v} for k, v in null.items()],
        "control_js": [{"n": k[0], "t": k[1], "js": v} for k, v in control_js.items()],
        "greedy": greedy,
    }
    results["hypotheses"] = evaluate_hypotheses(manifest, results)
    return results


def evaluate_hypotheses(manifest, res) -> list[dict]:
    H = manifest["hypotheses"]
    models = [r["id"] for r in manifest["respondents"] if not r.get("control")]
    pooled = {(p["respondent"], p["n"], p["t"]): p for p in res["pooled"]}
    nulls = {(x["n"], x["t"]): x for x in res["null"]}
    cross = {(x["n"], x["t"]): x for x in res["cross"]}
    head = manifest["headline_temperatures"]
    temps_hot = sorted(t for t in manifest["temperatures"] if t > 0)
    out = []

    def mean_ent(n, t):
        return float(np.mean([pooled[(m, n, t)]["normalized_entropy"] for m in models if (m, n, t) in pooled]))

    # H1
    thr = H["H1_max_mean_normalized_entropy"]
    detail = {f"r{n}_t{t:g}": round(mean_ent(n, t), 3) for n in manifest["ranges"] for t in temps_hot}
    below = all(mean_ent(n, t) < thr for n in manifest["ranges"] for t in head)
    monotone = all(mean_ent(n, temps_hot[0]) <= mean_ent(n, temps_hot[-1]) for n in manifest["ranges"])
    out.append({"id": "H1", "claim": f"Mean model normalized entropy < {thr} at headline temperatures, and lowest at the "
                f"lowest sampled temperature", "held": bool(below and monotone),
                "detail": {"mean_normalized_entropy": detail, "below_threshold": below, "lowest_at_low_T": monotone}})
    # H2
    t2 = H["H2_temperature"]
    modes = {m: pooled[(m, 10, t2)]["mode"] for m in models if (m, 10, t2) in pooled}
    n7 = sum(v == 7 for v in modes.values())
    out.append({"id": "H2", "claim": f"7 is the mode in 1-10 for a majority of model families at T={t2:g}",
                "held": n7 > len(modes) / 2, "detail": {"modes": modes, "families_with_mode_7": n7}})
    # H3
    alpha = H["H3_alpha"]
    d3 = {f"r{n}_t{t:g}": {"observed": round(nulls[(n, t)]["observed_mean_js"], 4),
                          "null_mean": round(nulls[(n, t)]["null_mean_js"], 4),
                          "p": nulls[(n, t)]["p_value_js"]}
          for n in manifest["ranges"] for t in head if (n, t) in nulls}
    out.append({"id": "H3", "claim": f"Mean pairwise JS far below the independent-bias null (one-sided p < {alpha}) "
                f"at every range and headline temperature",
                "held": all(v["p"] < alpha for v in d3.values()) and bool(d3), "detail": d3})
    # H4
    mult = H["H4_min_multiple_of_uniform"]
    d4 = {}
    for (n, t), c in cross.items():
        if t in head:
            loo = c["pooled"]["single_guess_hit_rate_global_loo"]
            d4[f"r{n}_t{t:g}"] = {"global_loo": round(loo, 4), "uniform": 1 / n, "multiple": round(loo * n, 2)}
    out.append({"id": "H4", "claim": f"Leave-one-model-out global single-guess hit rate >= {mult}x uniform at every "
                f"range and headline temperature", "held": all(v["multiple"] >= mult for v in d4.values()) and bool(d4),
                "detail": d4})
    return out


def cmd_sample(args):
    m = load_manifest(args.pilot)
    if not args.pilot:
        verify_lock(check_live=True)
    run_sampling(m, evidence_dir(args.pilot), args.respondents, workers=args.workers)


def cmd_score(args):
    m = load_manifest(args.pilot)
    if not args.pilot:
        verify_lock()
        m = json.loads(MANIFEST.read_text())
    ev = evidence_dir(args.pilot)
    rows = load_responses(ev)
    # Re-parse every stored raw response with the (frozen) parse rule; the value
    # recorded at sampling time is a convenience, not the scored answer.
    for r in rows:
        r["value"], r["status"] = parse_answer(r["text"], r["n"])
    try:
        res = score(m, rows)
        # Parse-rule sensitivity: rescore with the lenient rule; reported, never the headline.
        lenient = [{**r} for r in rows]
        for r in lenient:
            r["value"], r["status"] = parse_answer_lenient(r["text"], r["n"])
        alt = score(m, lenient)
        res["sensitivity_lenient_parse"] = {
            "null": alt["null"], "hypotheses": alt["hypotheses"],
            "parse_failure_rate": {p["respondent"]: p["parse_failure_rate"] for p in alt["pooled"]},
        }
    except integrity.IntegrityError as e:
        (ev / "INTEGRITY_FAILURE.txt").write_text(str(e))
        sys.exit(f"ABORT: integrity check failed, nothing scored.\n{e}")
    res["meta"] = {"pilot": args.pilot, "scored_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                   "n_responses": len(rows),
                   "manifest_sha256": sha256(MANIFEST.read_bytes()),
                   "lock": None if args.pilot else json.loads(LOCK.read_text())}
    (ev / "results.json").write_text(json.dumps(res, indent=1))
    for h in res["hypotheses"]:
        print(f"{h['id']}: {'HELD' if h['held'] else 'NOT HELD'}  {h['claim']}")


def cmd_report(args):
    from harness.report import write_report
    m = load_manifest(args.pilot)
    path = write_report(m, evidence_dir(args.pilot), ROOT / "reports", pilot=args.pilot)
    print(f"report: {path}")


def cmd_fingerprint(args):
    print(json.dumps(live_fingerprints(json.loads(MANIFEST.read_text())), indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["fingerprint", "freeze", "sample", "score", "report", "run"])
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--respondents", nargs="*")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    if args.cmd == "run":
        cmd_sample(args); cmd_score(args); cmd_report(args)
    else:
        {"fingerprint": cmd_fingerprint, "freeze": cmd_freeze, "sample": cmd_sample,
         "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    main()
