"""Query loop: retry/backoff, raw capture, resumable.

Every raw response goes to evidence/raw/<respondent>/<cell>.jsonl and to the
``responses`` table in evidence/run.sqlite. A rerun skips samples already stored,
so a crash or reboot costs nothing.
"""
from __future__ import annotations

import json
import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from harness.parse import parse_answer
from harness.probe import Cell, build_cells, sample_seed, samples_for
from respondents.base import make_respondent

SCHEMA = """
CREATE TABLE IF NOT EXISTS responses (
  respondent TEXT, n INTEGER, temperature REAL, prompt_id TEXT, sample_idx INTEGER,
  prompt TEXT, seed INTEGER, text TEXT, value INTEGER, status TEXT,
  latency_s REAL, ts REAL,
  PRIMARY KEY (respondent, n, temperature, prompt_id, sample_idx)
);
"""


def connect(evidence: Path) -> sqlite3.Connection:
    evidence.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(evidence / "run.sqlite")
    db.executescript(SCHEMA)
    return db


def _done(db, cell: Cell) -> set[int]:
    rows = db.execute(
        "SELECT sample_idx FROM responses WHERE respondent=? AND n=? AND temperature=? AND prompt_id=?",
        (cell.respondent, cell.n, cell.temperature, cell.prompt_id),
    )
    return {r[0] for r in rows}


def _ask(resp, cell: Cell, i: int, retries: int = 6):
    # Greedy repeats deliberately use no seed: the determinism guard is asking
    # whether temperature 0 alone is deterministic.
    seed = None if cell.greedy else sample_seed(cell, i)
    delay = 2.0
    for attempt in range(retries):
        t0 = time.time()
        try:
            text = resp.query(cell.prompt, cell.temperature, seed)
            return cell, i, seed, text, time.time() - t0
        except Exception as e:  # network/server hiccup: back off and retry
            if attempt == retries - 1:
                raise RuntimeError(f"{cell.key}#{i} failed after {retries} tries: {e}") from e
            time.sleep(delay)
            delay *= 2


def run_sampling(manifest: dict, evidence: Path, respondent_ids=None, workers: int = 4, log=print):
    db = connect(evidence)
    raw_dir = evidence / "raw"
    cells = build_cells(manifest, respondent_ids)
    by_resp: dict[str, list[Cell]] = {}
    for c in cells:
        by_resp.setdefault(c.respondent, []).append(c)
    specs = {r["id"]: r for r in manifest["respondents"]}

    for rid, rcells in by_resp.items():
        resp = make_respondent(specs[rid], manifest["max_tokens"])
        todo = []
        for c in rcells:
            done = _done(db, c)
            todo += [(c, i) for i in range(samples_for(c, manifest)) if i not in done]
        if not todo:
            log(f"[{rid}] complete")
            continue
        log(f"[{rid}] {len(todo)} samples to collect")
        t_start, n_done = time.time(), 0
        pool_size = 1 if resp.is_control else workers
        with ThreadPoolExecutor(pool_size) as ex:
            futs = [ex.submit(_ask, resp, c, i) for c, i in todo]
            for f in as_completed(futs):
                cell, i, seed, text, lat = f.result()
                value, status = parse_answer(text, cell.n)
                rec = {
                    "respondent": cell.respondent, "n": cell.n, "temperature": cell.temperature,
                    "prompt_id": cell.prompt_id, "sample_idx": i, "prompt": cell.prompt, "seed": seed,
                    "text": text, "value": value, "status": status, "latency_s": round(lat, 4),
                    "ts": time.time(),
                }
                db.execute(
                    "INSERT OR REPLACE INTO responses VALUES (:respondent,:n,:temperature,:prompt_id,:sample_idx,"
                    ":prompt,:seed,:text,:value,:status,:latency_s,:ts)",
                    rec,
                )
                d = raw_dir / cell.respondent
                d.mkdir(parents=True, exist_ok=True)
                with open(d / f"r{cell.n}_t{cell.temperature:g}_{cell.prompt_id}.jsonl", "a", encoding="utf-8") as fh:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_done += 1
                if n_done % 200 == 0:
                    db.commit()
                    rate = n_done / (time.time() - t_start)
                    log(f"[{rid}] {n_done}/{len(todo)}  {rate:.1f}/s  eta {(len(todo) - n_done) / rate / 60:.1f} min")
        db.commit()
        log(f"[{rid}] done in {(time.time() - t_start) / 60:.1f} min")
    db.close()


def load_responses(evidence: Path) -> list[dict]:
    db = sqlite3.connect(evidence / "run.sqlite")
    db.row_factory = sqlite3.Row
    rows = [dict(r) for r in db.execute("SELECT * FROM responses")]
    db.close()
    return rows
