"""Prompt construction and the sweep over (respondent, range, temperature, prompt) cells."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass(frozen=True)
class Cell:
    respondent: str
    n: int
    temperature: float
    prompt_id: str
    template: str

    @property
    def prompt(self) -> str:
        return self.template.format(N=self.n)

    @property
    def greedy(self) -> bool:
        return self.temperature == 0.0

    @property
    def key(self) -> str:
        return f"{self.respondent}|r{self.n}|t{self.temperature:g}|{self.prompt_id}"


def sample_seed(cell: Cell, i: int) -> int:
    """Per-sample seed derived from the cell and index, so reruns ask the same question the same way."""
    return int(hashlib.sha256(f"{cell.key}|{i}".encode()).hexdigest()[:8], 16)


def build_cells(manifest: dict, respondent_ids: list[str] | None = None) -> list[Cell]:
    cells = []
    for r in manifest["respondents"]:
        if respondent_ids and r["id"] not in respondent_ids:
            continue
        max_t = r.get("max_temperature")
        for n in manifest["ranges"]:
            for t in manifest["temperatures"]:
                if max_t is not None and t > max_t:
                    continue
                for p in manifest["prompts"]:
                    cells.append(Cell(r["id"], n, float(t), p["id"], p["template"]))
    return cells


def samples_for(cell: Cell, manifest: dict) -> int:
    return manifest["greedy_repeats"] if cell.greedy else manifest["samples_per_cell"]
