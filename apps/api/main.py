"""AgentGuard API service.

Exposes endpoints to trigger eval runs, fetch trajectories/results, and
serve regression reports to the dashboard.
"""

from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="AgentGuard API", version="0.1.0")


class HealthResponse(BaseModel):
    status: str = "ok"


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse()


@app.get("/runs")
def list_runs() -> list[dict]:
    """List past evaluation runs. Placeholder — wire up to a datastore."""
    return []


@app.get("/runs/{run_id}")
def get_run(run_id: str) -> dict:
    """Fetch a single run's results. Placeholder — wire up to a datastore."""
    return {"run_id": run_id, "status": "not_found"}
