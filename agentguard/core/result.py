"""Result types returned by evaluators and judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"


@dataclass
class EvalResult:
    """Outcome of running a single evaluator or judge against a trajectory."""

    name: str
    verdict: Verdict
    score: Optional[float] = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS


@dataclass
class RunResult:
    """Aggregated results for a single trajectory across all evaluators."""

    trajectory_id: str
    agent_name: str
    scenario_name: str
    eval_results: list[EvalResult] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return all(r.passed for r in self.eval_results)

    @property
    def pass_rate(self) -> float:
        if not self.eval_results:
            return 0.0
        return sum(1 for r in self.eval_results if r.passed) / len(self.eval_results)

    def summary(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "agent_name": self.agent_name,
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "results": [
                {"name": r.name, "verdict": r.verdict.value, "score": r.score}
                for r in self.eval_results
            ],
        }
