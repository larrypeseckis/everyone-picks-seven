"""Local open-weight families served by Ollama.

Fingerprint = model tag + content digest + quantization + parameter size +
Ollama server version. Thinking is disabled (``think: false``) so reasoning
models answer the probe directly, like a deployed chat endpoint would.
"""
from __future__ import annotations

import json
import os
import urllib.request

from respondents.base import Respondent

HOST = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")


def _post(path: str, body: dict, timeout: float = 300) -> dict:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


def _get(path: str) -> dict:
    with urllib.request.urlopen(HOST + path, timeout=30) as r:
        return json.loads(r.read())


class OllamaRespondent(Respondent):
    def query(self, prompt, temperature, seed=None):
        options = {"temperature": temperature, "num_predict": self.max_tokens}
        if seed is not None:
            options["seed"] = seed
        body = {
            "model": self.spec["model"],
            "messages": [{"role": "user", "content": prompt}],
            "stream": False,
            "think": False,
            "options": options,
        }
        return _post("/api/chat", body)["message"]["content"]

    def fingerprint(self):
        tags = {m["name"]: m for m in _get("/api/tags")["models"]}
        tag = tags.get(self.spec["model"])
        if tag is None:
            raise RuntimeError(f"model {self.spec['model']} not pulled in Ollama")
        show = _post("/api/show", {"model": self.spec["model"]}, timeout=60)
        details = show.get("details", {})
        return {
            "provider": "ollama",
            "model": self.spec["model"],
            "digest": tag["digest"],
            "size_bytes": tag["size"],
            "family": details.get("family"),
            "parameter_size": details.get("parameter_size"),
            "quantization": details.get("quantization_level"),
            "runtime": f"ollama {_get('/api/version')['version']}",
        }
