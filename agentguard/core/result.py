"""Result types returned by evaluators and judges."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    PASS = "pass"
    FAIL = "fail"
    WARN = "warn"
    ERROR = "error"


class FailureCategory(str, Enum):
    """The five broad buckets a FailureType belongs to."""

    TOOL_FAILURE = "tool_failure"
    AGENT_FAILURE = "agent_failure"
    CONTRACT_FAILURE = "contract_failure"
    SAFETY_FAILURE = "safety_failure"
    QUALITY_FAILURE = "quality_failure"


class FailureType(str, Enum):
    """A structured classification of *why* a run failed.

    Grouped into five categories (see FAILURE_CATEGORY below for the
    mapping). The point of this taxonomy is that "FAIL" alone tells a
    developer nothing actionable — "CONTRACT_FAILURE / MAX_CALLS_EXCEEDED
    on lookup_order, expected 1 call, observed 2" tells them exactly where
    to look.
    """

    # TOOL_FAILURE — the tool itself misbehaved (often chaos-injected)
    TIMEOUT = "timeout"
    HTTP_ERROR = "http_error"
    MALFORMED_RESPONSE = "malformed_response"
    RATE_LIMIT = "rate_limit"

    # AGENT_FAILURE — the agent chose the wrong action
    WRONG_TOOL = "wrong_tool"
    WRONG_ARGUMENT = "wrong_argument"
    TOOL_LOOP = "tool_loop"
    PREMATURE_TERMINATION = "premature_termination"

    # CONTRACT_FAILURE — a declared behavioral contract was violated
    FORBIDDEN_TOOL = "forbidden_tool"
    MISSING_REQUIRED_STEP = "missing_required_step"
    MAX_CALLS_EXCEEDED = "max_calls_exceeded"
    ORDERING_VIOLATION = "ordering_violation"
    MAX_TURNS_EXCEEDED = "max_turns_exceeded"

    # SAFETY_FAILURE — the agent did something actively unsafe
    PII_LEAK = "pii_leak"
    POLICY_VIOLATION = "policy_violation"
    UNSAFE_ACTION = "unsafe_action"

    # QUALITY_FAILURE — the agent's output was substantively wrong
    WRONG_FINAL_ANSWER = "wrong_final_answer"
    INCOMPLETE_RESPONSE = "incomplete_response"
    HALLUCINATION = "hallucination"


FAILURE_CATEGORY: dict[FailureType, FailureCategory] = {
    FailureType.TIMEOUT: FailureCategory.TOOL_FAILURE,
    FailureType.HTTP_ERROR: FailureCategory.TOOL_FAILURE,
    FailureType.MALFORMED_RESPONSE: FailureCategory.TOOL_FAILURE,
    FailureType.RATE_LIMIT: FailureCategory.TOOL_FAILURE,
    FailureType.WRONG_TOOL: FailureCategory.AGENT_FAILURE,
    FailureType.WRONG_ARGUMENT: FailureCategory.AGENT_FAILURE,
    FailureType.TOOL_LOOP: FailureCategory.AGENT_FAILURE,
    FailureType.PREMATURE_TERMINATION: FailureCategory.AGENT_FAILURE,
    FailureType.FORBIDDEN_TOOL: FailureCategory.CONTRACT_FAILURE,
    FailureType.MISSING_REQUIRED_STEP: FailureCategory.CONTRACT_FAILURE,
    FailureType.MAX_CALLS_EXCEEDED: FailureCategory.CONTRACT_FAILURE,
    FailureType.ORDERING_VIOLATION: FailureCategory.CONTRACT_FAILURE,
    FailureType.MAX_TURNS_EXCEEDED: FailureCategory.CONTRACT_FAILURE,
    FailureType.PII_LEAK: FailureCategory.SAFETY_FAILURE,
    FailureType.POLICY_VIOLATION: FailureCategory.SAFETY_FAILURE,
    FailureType.UNSAFE_ACTION: FailureCategory.SAFETY_FAILURE,
    FailureType.WRONG_FINAL_ANSWER: FailureCategory.QUALITY_FAILURE,
    FailureType.INCOMPLETE_RESPONSE: FailureCategory.QUALITY_FAILURE,
    FailureType.HALLUCINATION: FailureCategory.QUALITY_FAILURE,
}


def category_of(failure_type: FailureType) -> FailureCategory:
    return FAILURE_CATEGORY[failure_type]


@dataclass
class Evidence:
    """One concrete, inspectable fact backing a verdict.

    A single EvalResult can carry several of these — e.g. a contract
    violation with both a forbidden-tool-call fact and a missing-step
    fact. Each Evidence entry is independently actionable: it names what
    kind of failure it is (if any — PASS-supporting evidence has no
    failure_type), where in the trajectory it happened, and what was
    expected vs. observed.
    """

    failure_type: FailureType | None = None
    step_index: int | None = None
    message: str = ""
    expected: Any = None
    actual: Any = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "failure_type": self.failure_type.value if self.failure_type else None,
            "category": category_of(self.failure_type).value if self.failure_type else None,
            "step_index": self.step_index,
            "message": self.message,
            "expected": self.expected,
            "actual": self.actual,
        }


@dataclass
class EvalResult:
    """Outcome of running a single evaluator or judge against a trajectory."""

    name: str
    verdict: Verdict
    score: float | None = None
    reason: str = ""
    details: dict[str, Any] = field(default_factory=dict)
    evidence: list[Evidence] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict == Verdict.PASS

    @property
    def failure_types(self) -> list[FailureType]:
        """Every distinct FailureType this result's evidence points to."""
        seen: list[FailureType] = []
        for e in self.evidence:
            if e.failure_type is not None and e.failure_type not in seen:
                seen.append(e.failure_type)
        return seen

    @property
    def primary_failure_type(self) -> FailureType | None:
        """The first classified failure, for callers that just want one
        headline reason rather than the full evidence list."""
        types = self.failure_types
        return types[0] if types else None


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

    @property
    def all_evidence(self) -> list[Evidence]:
        return [e for r in self.eval_results for e in r.evidence]

    def summary(self) -> dict[str, Any]:
        return {
            "trajectory_id": self.trajectory_id,
            "agent_name": self.agent_name,
            "scenario_name": self.scenario_name,
            "passed": self.passed,
            "pass_rate": self.pass_rate,
            "results": [
                {
                    "name": r.name,
                    "verdict": r.verdict.value,
                    "score": r.score,
                    "evidence": [e.as_dict() for e in r.evidence],
                }
                for r in self.eval_results
            ],
        }
