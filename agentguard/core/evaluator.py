"""Evaluator base class and orchestration logic.

An Evaluator takes a completed Trajectory (optionally alongside the Contract
it should satisfy) and produces one or more EvalResults. Concrete evaluators
live in agentguard.judges (llm_judge, rule_judge) or can be defined by users.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from agentguard.contracts.schema import Contract
from agentguard.core.result import EvalResult, RunResult
from agentguard.core.trajectory import Trajectory


class Evaluator(ABC):
    """Base class for anything that scores a trajectory."""

    name: str = "unnamed_evaluator"

    @abstractmethod
    def evaluate(
        self, trajectory: Trajectory, contract: Optional[Contract] = None
    ) -> EvalResult:
        """Score a single trajectory, optionally against a contract."""
        raise NotImplementedError


class EvaluationPipeline:
    """Runs a set of evaluators against a trajectory and aggregates results."""

    def __init__(self, evaluators: list[Evaluator]):
        self.evaluators = evaluators

    def run(
        self, trajectory: Trajectory, contract: Optional[Contract] = None
    ) -> RunResult:
        results = [ev.evaluate(trajectory, contract) for ev in self.evaluators]
        return RunResult(
            trajectory_id=trajectory.id,
            agent_name=trajectory.agent_name,
            scenario_name=trajectory.scenario_name,
            eval_results=results,
        )
