# everyone-picks-seven

A harness for measuring **correlated predictability** across language-model families, using the cheapest clean probe there is: "pick a random number between 1 and N."

## The two things that must stay separate

- **Collapse** is within-model: how far one model's answer distribution sits from uniform. A model can be strongly collapsed and still be no security problem if every model collapses somewhere different.
- **Correlated collapse** is across-model: how much the collapsed distributions agree with each other. This is the security property. It is what removes the independence that redundancy depends on.

The harness exists to measure the second and to show it is not an artifact of the first. The decisive comparison is against an **independent-bias null**: each model keeps its realized distribution exactly (its entropy, its degree of collapse) but that shape is moved onto a random relabeling of the values, independently per model. That destroys only the cross-model alignment. The finding is the gap between observed mean pairwise Jensen-Shannon divergence and that null.

The operational translation is the **single-guess hit rate**: one fixed global guess against a deployment whose model you do not know, compared with the per-model oracle and with 1/N.

See [DESIGN.md](DESIGN.md) for the full spec and the reasoning behind it.

## v1 scope

Open-weight families only, run locally under Ollama on one RTX 4090. The closed-model adapters (`respondents/openai.py`, `anthropic.py`, `google.py`) are written but **untested**: no API keys were available for v1.

| id | family | Ollama tag |
|---|---|---|
| llama3.1-8b | Meta Llama | llama3.1:8b |
| qwen3-8b | Alibaba Qwen | qwen3:8b |
| mistral-7b | Mistral | mistral:7b |
| gemma4-8b | Google Gemma | gemma4:latest |
| phi4-14b | Microsoft Phi | phi4:14b |
| prng | uniform control | Python `random.Random` |

Exact digests, quantization and runtime version are pinned in `manifest.json` under `fingerprints` at freeze time. (Ollama reports Mistral's architecture family as `llama`; that is the architecture label, the weights are Mistral's.)

Sweep: ranges {10, 20, 50, 100} × temperatures {0, 0.7, 1.0, 1.5} × 5 prompt paraphrases. 200 samples per cell at T > 0. At T = 0, 5 repeats per cell: that is the greedy answer plus the determinism guard. Headline temperatures are 0.7 and 1.0.

## Running it

```bash
pip install numpy matplotlib pytest
python -m pytest -q                              # parser, metrics, null-model sanity tests
python -m harness.controller run --pilot         # small unfrozen pilot -> evidence/pilot, reports/*_pilot_summary.md
python -m harness.controller freeze              # pins fingerprints, hashes manifest + scoring code -> manifest.lock
python -m harness.controller sample              # canonical run; refuses if lock or fingerprints changed
python -m harness.controller score               # integrity checks, then metrics, null, hypotheses -> evidence/results.json
python -m harness.controller report              # reports/<timestamp>_summary.md + figures
```

Sampling is resumable: every response is committed to `evidence/run.sqlite` and appended to `evidence/raw/<respondent>/<cell>.jsonl`. A rerun of `sample` skips what is already there.

## Pre-registration discipline

`freeze` writes the live model fingerprints, the parse rule and the metric definitions into `manifest.json`, then records SHA-256 hashes of the manifest **and of the scoring code** (`parse.py`, `metrics.py`, `null_model.py`, `integrity.py`, `probe.py`, `controller.py`) in `manifest.lock`. `sample` and `score` refuse to run if any of them changed. Any refreeze must be justified in `AMENDMENTS.md`.

Integrity checks run before anything is scored, and a failure aborts the run:
- every cell holds exactly the predeclared number of responses (parse failures included, nothing dropped);
- the PRNG control scores normalized entropy ≥ 0.95 everywhere with zero parse failures, and at the headline temperatures it is a JS outlier from every model.

The determinism guard at T = 0 is a soft check. It is flagged in the report, not fatal.

## Honest limitations

- One narrow generative choice. This harness does **not** measure whether model judgment, classification or triage converges. That argument rests on the separate literature on cross-model transferability, not on these runs.
- The shared mode most likely comes from a shared human prior and overlapping training data, not from shared weights. Correlated predictability bites regardless of cause.
- Results are version-pinned and perishable. v1 uses 4-bit (Q4_K_M) quantized weights of ~8B-14B models. Full-precision or larger checkpoints may differ.
- Parse failures count as misses in hit rates and as mass outside the distribution in `p_mode`/`top3_mass`. The distributions used for entropy and JS are over parsed answers only.
- Refusals are real data. In the pilot Phi-4 declined to pick about a third of the time ("I can't generate truly random numbers, but here's how..."). Those are parse failures, so they count as misses for the attacker and are excluded from Phi's distribution. Mistral's strict-rule failures (~12%) are mostly a number followed by trivia about other numbers. A lenient-rule rescoring is reported alongside the headline.
- Temperature 0 was not perfectly deterministic in the pilot (1-7 of 20 cells per model differed across repeats). The likely cause is batched decoding of concurrent requests in Ollama, but that is not verified. The report footnotes how many cells per model differed across repeats. The greedy answer is the modal greedy response.
- Logprob mode (spec: optional) is not implemented in v1.
- Cost: v1 is local, so there is no API spend. Wall-clock time is reported in the run log.
