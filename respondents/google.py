"""Google Gemini generateContent. Requires GEMINI_API_KEY. Untested in v1 (no key on the run machine)."""
from __future__ import annotations

import json
import os
import urllib.request

from respondents.base import Respondent


class GoogleRespondent(Respondent):
    def query(self, prompt, temperature, seed=None):
        cfg = {"temperature": temperature, "maxOutputTokens": self.max_tokens}
        if seed is not None:
            cfg["seed"] = seed
        body = {"contents": [{"role": "user", "parts": [{"text": prompt}]}], "generationConfig": cfg}
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.spec['model']}:generateContent"
        )
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json", "x-goog-api-key": os.environ["GEMINI_API_KEY"]},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read())
        parts = d["candidates"][0].get("content", {}).get("parts", [])
        return "".join(p.get("text", "") for p in parts)

    def fingerprint(self):
        return {"provider": "google", "model": self.spec["model"]}
