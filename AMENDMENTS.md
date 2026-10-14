# Amendments

Any change to the frozen manifest or scoring code after `freeze` is recorded here, with the date, the reason, and whether any canonical data had been seen at that point.

## Pre-freeze design changes driven by the pilot (2026-10-13, no canonical data collected)

The pilot (20 samples/cell, `evidence/pilot/`) exists to break the pipeline before the canonical run. It did, and these changes went in before `freeze`:

1. **Parser: ellipsis.** `"...5!"` was a `no_number` failure: the lookbehind meant to reject `.5` decimals also rejected an ellipsis. Fixed and tested.
2. **Parser: bold answer.** When several numbers appear but exactly one distinct number is in markdown bold (`**17**`), that is the answer. Models bold their pick and then add trivia about it ("it's prime, 1 + 7 = 8").
3. **Parser: list enumerators.** Line-initial `1. ` / `2) ` are removed before extraction. Phi-4 refuses about a third of the time and returns a numbered how-to list. Those must be failures, not answers of "1".
4. **Scoring re-parses raw text** with the frozen rule. The value stored at sampling time is never what gets scored.
5. **Parse-rule sensitivity check added.** The headline is rescored with a lenient rule (first in-range integer). Mistral's strict-rule failure rate is ~12%, so the result must be shown not to depend on the rule.
6. **Uniform-control entropy check.** The fixed 0.95 threshold was wrong. A true uniform source with 100 samples over 100 values scores ~0.92 Miller-Madow normalized entropy. It is replaced by the 0.1th percentile of a simulated true-uniform source at the same sample size.
7. **Uniform-control outlier check.** The original rule (control-model JS > mean model-model JS) compared per-prompt against pooled JS. It was also invalid in principle: models collapsed onto *different* values legitimately diverge from each other more than from uniform. It is replaced by: control-model JS exceeds the 99.9th percentile of uniform-vs-uniform JS at matched sample sizes, for every model. The old comparison is still reported.
8. **Bootstrap CI for entropy** re-centered on the point estimate. The parametric bootstrap from the plug-in distribution is biased low, and its interval did not contain the estimate for near-uniform sources.
9. **Determinism guard** excludes the PRNG control.
