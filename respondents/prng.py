"""The uniform control: a real PRNG wearing the respondent interface.

It reads N out of the prompt and answers a uniform integer in 1..N. If the
pipeline does not score this as near-uniform and as an outlier from every
model, the pipeline is wrong and the run aborts (see harness/integrity.py).
"""
from __future__ import annotations

import random
import re

from respondents.base import Respondent


class PRNGRespondent(Respondent):
    def query(self, prompt, temperature, seed=None):
        n = int(re.findall(r"\d+", prompt)[-1])
        rng = random.Random(seed)  # seeded per sample so the control is reproducible too
        return str(rng.randint(1, n))

    def fingerprint(self):
        return {"provider": "prng", "model": "python random.Random (Mersenne Twister), seeded per sample"}
