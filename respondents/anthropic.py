"""Anthropic Messages API. Requires ANTHROPIC_API_KEY. Untested in v1 (no key on the run machine).

The API caps temperature at 1.0, so temperatures above that are skipped for this
respondent (the sampler records them as not-applicable rather than clamping).
"""
from __future__ import annotations

import json
import os
import urllib.request

from respondents.base import Respondent

MAX_TEMPERATURE = 1.0


class AnthropicRespondent(Respondent):
    max_temperature = MAX_TEMPERATURE

    def query(self, prompt, temperature, seed=None):
        body = {
            "model": self.spec["model"],
            "max_tokens": self.max_tokens,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        req = urllib.request.Request(
            "https://api.anthropic.com/v1/messages",
            data=json.dumps(body).encode(),
            headers={
                "Content-Type": "application/json",
                "x-api-key": os.environ["ANTHROPIC_API_KEY"],
                "anthropic-version": "2023-06-01",
            },
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        return "".join(b.get("text", "") for b in d["content"] if b["type"] == "text")

    def fingerprint(self):
        return {"provider": "anthropic", "model": self.spec["model"]}
