"""Summary markdown, JS heatmaps, entropy-vs-temperature curves, per-model answer distributions."""
from __future__ import annotations

import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


def _fmt(x, d=3):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def heatmap(cross, ids_all, control_js, n, t, path):
    ids = cross["pooled"]["models"]
    labels = ids + ["prng"]
    k = len(labels)
    M = np.zeros((k, k))
    for key, v in cross["pooled"]["pairwise_js"].items():
        a, b = key.split("~")
        i, j = labels.index(a), labels.index(b)
        M[i, j] = M[j, i] = v
    for m, v in control_js.items():
        i = labels.index(m)
        M[i, k - 1] = M[k - 1, i] = v
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis_r")
    ax.set_xticks(range(k), labels, rotation=40, ha="right")
    ax.set_yticks(range(k), labels)
    for i in range(k):
        for j in range(k):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > 0.5 else "black")
    ax.set_title(f"Pairwise JS divergence (bits), 1-{n}, T={t:g}\n0 = identical, 1 = disjoint")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def entropy_curves(res, manifest, path):
    pooled = res["pooled"]
    ranges = manifest["ranges"]
    fig, axes = plt.subplots(1, len(ranges), figsize=(4 * len(ranges), 3.6), sharey=True)
    for ax, n in zip(np.atleast_1d(axes), ranges):
        for rid in sorted({p["respondent"] for p in pooled}):
            pts = sorted((p["t"], p["normalized_entropy"], *p["normalized_entropy_ci"])
                         for p in pooled if p["respondent"] == rid and p["n"] == n)
            if not pts:
                continue
            t, h, lo, hi = map(np.array, zip(*pts))
            ax.errorbar(t, h, yerr=[h - lo, hi - h], marker="o", capsize=3, label=rid,
                        linestyle="--" if rid == manifest["control_id"] else "-")
        ax.set_title(f"1-{n}")
        ax.set_xlabel("temperature")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].set_ylabel("normalized entropy (Miller-Madow)")
    handles, labels = np.atleast_1d(axes)[-1].get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="center left", bbox_to_anchor=(1.0, 0.5))
    fig.suptitle("Within-model collapse vs temperature (1 = uniform)")
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def distributions(res, n, t, path):
    rows = [p for p in res["pooled_counts"] if p["n"] == n and p["t"] == t]
    rows.sort(key=lambda p: p["respondent"])
    fig, axes = plt.subplots(len(rows), 1, figsize=(max(6, n * 0.12 + 4), 1.3 * len(rows) + 0.6), sharex=True)
    for ax, p in zip(np.atleast_1d(axes), rows):
        c = np.array(p["counts"], float)
        ax.bar(np.arange(1, n + 1), c / max(c.sum(), 1), width=0.85)
        ax.axhline(1 / n, color="red", lw=0.8, ls=":")
        ax.set_ylabel(p["respondent"], rotation=0, ha="right", va="center", fontsize=8)
        ax.tick_params(labelsize=7)
    np.atleast_1d(axes)[-1].set_xlabel("answer (dotted line = uniform)")
    fig.suptitle(f"Answer distributions, 1-{n}, T={t:g}, pooled over prompts")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_report(manifest, evidence: Path, reports: Path, pilot=False) -> Path:
    res = json.loads((evidence / "results.json").read_text())
    stamp = time.strftime("%Y%m%d-%H%M%S") + ("_pilot" if pilot else "")
    figdir = reports / f"{stamp}_figs"
    figdir.mkdir(parents=True, exist_ok=True)
    head = manifest["headline_temperatures"]
    cross = {(c["n"], c["t"]): c for c in res["cross"]}
    nulls = {(c["n"], c["t"]): c for c in res["null"]}
    cjs = {(c["n"], c["t"]): c["js"] for c in res["control_js"]}
    models = [r["id"] for r in manifest["respondents"] if not r.get("control")]

    for (n, t), c in cross.items():
        heatmap(c, models, cjs.get((n, t), {}), n, t, figdir / f"js_r{n}_t{t:g}.png")
    entropy_curves(res, manifest, figdir / "entropy_vs_temperature.png")
    for n in manifest["ranges"]:
        distributions(res, n, 1.0, figdir / f"dist_r{n}_t1.png")

    L = []
    L.append(f"# Everyone Picks Seven: {'PILOT ' if pilot else ''}summary ({stamp})\n")
    if pilot:
        L.append("> **Pilot run.** Small samples, unfrozen manifest. Pipeline check only: not a finding.\n")
    meta = res.get("meta", {})
    L.append(f"Responses scored: {meta.get('n_responses')}. Manifest sha256: `{meta.get('manifest_sha256')}`.\n")
    L.append("**Collapse** is within-model: how far one model's answer distribution sits from uniform. "
             "**Correlated collapse** is across-model: how much the collapsed distributions agree with each other. "
             "The independent-bias null keeps each model's collapse exactly and destroys only the cross-model "
             "alignment, so the gap between observed and null is the correlated part alone.\n")

    L.append("## Headline: correlated collapse vs the independent-bias null\n")
    L.append("| range | T | mean pairwise JS (observed) | null mean JS | null 1st pct | p (one-sided) | "
             "shared mode rate (obs / null) | global guess | hit global (LOO) | hit oracle | uniform |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for n in manifest["ranges"]:
        for t in sorted(t for t in manifest["temperatures"] if t > 0):
            if (n, t) not in cross:
                continue
            c, z = cross[(n, t)]["pooled"], nulls[(n, t)]
            bold = "**" if t in head else ""
            L.append(f"| {bold}1-{n}{bold} | {bold}{t:g}{bold} | {_fmt(z['observed_mean_js'])} | {_fmt(z['null_mean_js'])} | "
                     f"{_fmt(z['null_js_p01'])} | {z['p_value_js']:.4f} | {_fmt(z['observed_shared_mode_rate'], 2)} / "
                     f"{_fmt(z['null_shared_mode_rate'], 2)} | {c['global_guess']} | "
                     f"{_fmt(c['single_guess_hit_rate_global'])} ({_fmt(c['single_guess_hit_rate_global_loo'])}) | "
                     f"{_fmt(c['single_guess_hit_rate_oracle'])} | {_fmt(c['uniform_baseline'])} |")
    L.append("\nObserved JS and the null are means over the five prompts (per-prompt distributions, permuted "
             "independently per model and per prompt). Hit rates use distributions pooled over prompts; "
             "LOO = the global guess is chosen from the other models and scored on the held-out one. "
             f"p-values are bounded below by 1/(reps+1) = {1 / (manifest['null_reps'] + 1):.4f}.\n")

    L.append("## Predeclared hypotheses\n")
    for h in res["hypotheses"]:
        L.append(f"- **{h['id']} {'HELD' if h['held'] else 'NOT HELD'}**: {h['claim']}")
        L.append(f"  - `{json.dumps(h['detail'])}`")
    L.append("")

    sens = res.get("sensitivity_lenient_parse")
    if sens:
        L.append("## Parse-rule sensitivity (lenient: first in-range integer)\n")
        L.append("| range | T | observed JS (strict / lenient) | null JS (strict / lenient) | p (strict / lenient) |")
        L.append("|---|---|---|---|---|")
        snull = {(x["n"], x["t"]): x for x in sens["null"]}
        for (n, t), z in sorted(nulls.items()):
            s = snull.get((n, t))
            if s:
                L.append(f"| 1-{n} | {t:g} | {z['observed_mean_js']:.3f} / {s['observed_mean_js']:.3f} | "
                         f"{z['null_mean_js']:.3f} / {s['null_mean_js']:.3f} | {z['p_value_js']:.4f} / {s['p_value_js']:.4f} |")
        L.append("\nHypotheses under the lenient rule: " + ", ".join(
            f"{h['id']} {'held' if h['held'] else 'NOT held'}" for h in sens["hypotheses"]) + "\n")

    L.append("## Within-model collapse (pooled over prompts)\n")
    L.append("| respondent | range | T | norm. entropy [95% CI] | mode | p_mode | top3 | top3 mass | parse fail |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for p in sorted(res["pooled"], key=lambda p: (p["n"], p["t"], p["respondent"])):
        lo, hi = p["normalized_entropy_ci"]
        L.append(f"| {p['respondent']} | 1-{p['n']} | {p['t']:g} | {_fmt(p['normalized_entropy'])} "
                 f"[{_fmt(lo)}, {_fmt(hi)}] | {p['mode']} | {_fmt(p['p_mode'])} | {p['top3']} | "
                 f"{_fmt(p['top3_mass'])} | {_fmt(p['parse_failure_rate'])} |")
    L.append("")

    L.append("## Per-prompt spread of cross-model agreement\n")
    L.append("| range | T | mean JS: mean ± sd [min, max] over prompts | shared mode rate: mean [min, max] |")
    L.append("|---|---|---|---|")
    for (n, t), c in sorted(cross.items()):
        a = c["aggregate_over_prompts"]
        js, sm = a["mean_pairwise_js"], a["shared_mode_rate"]
        L.append(f"| 1-{n} | {t:g} | {js['mean']:.3f} ± {js['sd']:.3f} [{js['min']:.3f}, {js['max']:.3f}] | "
                 f"{sm['mean']:.2f} [{sm['min']:.2f}, {sm['max']:.2f}] |")
    L.append("")

    L.append("## Greedy answers (T = 0)\n")
    L.append("| range | prompt | " + " | ".join(models) + " | most common | share |")
    L.append("|---|---|" + "---|" * len(models) + "---|---|")
    for g in res["greedy"]:
        L.append(f"| 1-{g['n']} | {g['prompt_id']} | " + " | ".join(str(g["modes"].get(m)) for m in models)
                 + f" | {g['most_common']} | {g['share_most_common']:.2f} |")
    L.append("")

    integ = res["integrity"]
    L.append("## Integrity\n")
    L.append(f"- Sample counts: {'OK' if integ['sample_counts']['ok'] else 'FAIL'} "
             f"({integ['sample_counts']['cells']} cells, exact predeclared counts).")
    u = integ["control"]["uniform"]
    L.append(f"- Uniform control: {sum(x['ok'] for x in u)}/{len(u)} (range, T) cells pass (entropy at or above "
             f"the simulated true-uniform 0.1th percentile at the same sample size; zero parse failures). "
             f"Lowest: {min(x['normalized_entropy'] for x in u):.3f}.")
    for o in integ["control"]["outlier"]:
        ceil = max(v["uniform_vs_uniform_p999"] for v in o["per_model"].values())
        L.append(f"  - 1-{o['n']} T={o['t']:g}: min JS(control, model) {o['min_js_control_to_model']:.3f} vs "
                 f"uniform-vs-uniform 99.9th pct {ceil:.3f}: {'separated from every model' if o['ok'] else 'NOT separated'}"
                 f"{'' if o['enforced'] else ' (not enforced at this T)'}; mean model-model JS (pooled) "
                 f"{o['mean_model_model_js_pooled']:.3f}")
    d = integ["determinism_t0"]
    L.append(f"- Determinism at T=0 (cells with differing repeats / cells): {d['cells_nondeterministic']}. "
             f"Non-deterministic respondents: {d['nondeterministic_respondents'] or 'none'}.")
    L.append(f"- Parse failure reasons: `{json.dumps(res['failure_reasons'])}`\n")

    L.append("## Figures\n")
    for f in sorted(figdir.iterdir()):
        L.append(f"![{f.stem}]({figdir.name}/{f.name})")
    L.append("")
    L.append("## Boundary\n")
    L.append("This measures correlated predictability in one toy generative choice. It does not measure whether "
             "model judgment, classification or triage converges; that argument rests on the cited literature on "
             "cross-model transferability, not on these runs.")

    out = reports / f"{stamp}_summary.md"
    out.write_text("\n".join(L), encoding="utf-8")
    return out
