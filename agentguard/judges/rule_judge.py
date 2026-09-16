"""Deterministic, rule-based judge.

Applies simple, explicit rules (regex/keyword checks, structural checks on
the trajectory) rather than relying on an LLM. Fast, cheap, and
deterministic — best for hard safety/format constraints.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from agentguard.contracts.schema import Contract
from agentguard.core.evaluator import Evaluator
from agentguard.core.result import EvalResult, Verdict
from agentguard.core.trajectory import StepType, Trajectory


@dataclass
class RuleJudge(Evaluator):
    """Checks a trajectory's text content against forbidden phrases/patterns."""

    name: str = "rule_judge"
    forbidden_patterns: list[str] = field(default_factory=list)

    def evaluate(
        self, trajectory: Trajectory, contract: Contract | None = None
    ) -> EvalResult:
        patterns = list(self.forbidden_patterns)
        if contract:
            patterns.extend(contract.forbidden_phrases)

        text_steps = [
            str(s.content) for s in trajectory.steps if s.type == StepType.MESSAGE
        ]
        full_text = "\n".join(text_steps)

        violations = [p for p in patterns if re.search(p, full_text, re.IGNORECASE)]

        verdict = Verdict.FAIL if violations else Verdict.PASS
        return EvalResult(
            name=self.name,
            verdict=verdict,
            reason=(
                f"Matched forbidden pattern(s): {violations}"
                if violations
                else "No forbidden patterns matched"
            ),
            details={"violations": violations},
        )
