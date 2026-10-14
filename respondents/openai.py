"""OpenAI chat completions. Requires OPENAI_API_KEY. Untested in v1 (no key on the run machine).

Pin a dated snapshot in the manifest (e.g. "gpt-4.1-2025-04-14"), not an alias.
Some reasoning models reject a temperature parameter; leave those out of the sweep.
"""
from __future__ import annotations

import json
import os
import urllib.request

from respondents.base import Respondent


class OpenAIRespondent(Respondent):
    def query(self, prompt, temperature, seed=None):
        body = {
            "model": self.spec["model"],
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "max_tokens": self.max_tokens,
        }
        if seed is not None:
            body["seed"] = seed
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        self._last_system_fingerprint = d.get("system_fingerprint")
        return d["choices"][0]["message"]["content"] or ""

    def fingerprint(self):
        return {"provider": "openai", "model": self.spec["model"], "snapshot": self.spec.get("snapshot", self.spec["model"])}
