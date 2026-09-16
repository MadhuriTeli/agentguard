"""AgentGuard API service.

Exposes endpoints to fetch trajectories/results and serve regression
reports to the dashboard.

There's no database yet — this reads the same JSON results file that
`scripts/run_evals.py --out ...` writes (see RESULTS_DIR below). That's an
honest placeholder, not a full persistence layer: results aren't queued,
versioned, or associated with a specific commit/branch. Swap
`_load_all_results` for a real datastore query once you need any of that.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

app = FastAPI(title="AgentGuard API", version="0.1.0")

RESULTS_DIR = Path(os.environ.get("AGENTGUARD_RESULTS_DIR", "results"))


class HealthResponse(BaseModel):
    status: str = "ok"


def _load_all_results() -> list[dict]:
    """Load every *_results.json file in RESULTS_DIR into one flat list."""
    if not RESULTS_DIR.exists():
        return []
    all_results: list[dict] = []
    for path in sorted(RESULTS_DIR.glob("*_results.json")):
        try:
            all_results.extend(json.loads(path.read_text()))
        except json.JSONDecodeError:
            continue  # skip malformed/partial files rather than 500ing
    return all_results


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/runs")
def list_runs() -> list[dict]:
    """List all recorded scenario runs found under RESULTS_DIR."""
    return _load_all_results()


@app.get("/runs/{trajectory_id}")
def get_run(trajectory_id: str) -> dict:
    """Fetch a single run's results by trajectory_id."""
    for run in _load_all_results():
        if run.get("trajectory_id") == trajectory_id:
            return run
    raise HTTPException(status_code=404, detail=f"No run found with id {trajectory_id}")
