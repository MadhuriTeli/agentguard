"""Trajectory-level evaluator for AI agent behavior."""

from __future__ import annotations

from typing import Any

from agentguard.core.evaluator import Evaluator
from agentguard.core.result import EvalResult, Verdict
from agentguard.core.trajectory import StepType, Trajectory


class TrajectoryJudge(Evaluator):
    """Validates expected tool calls and final agent responses."""

    name = "trajectory_judge"

    def __init__(
        self,
        expected_tool_calls: list[dict[str, Any]] | None = None,
        final_response_contains: list[str] | None = None,
    ):
        self.expected_tool_calls = expected_tool_calls or []
        self.final_response_contains = final_response_contains or []

    def evaluate(
        self,
        trajectory: Trajectory,
        contract=None,
    ) -> EvalResult:
        violations: list[str] = []

        actual_tool_calls = [
            step.content
            for step in trajectory.steps
            if step.type == StepType.TOOL_CALL
            and isinstance(step.content, dict)
        ]

        # ---------------------------------------------------------
        # Validate expected tool calls in order
        # ---------------------------------------------------------
        if self.expected_tool_calls:
            if len(actual_tool_calls) < len(self.expected_tool_calls):
                violations.append(
                    f"Expected at least {len(self.expected_tool_calls)} "
                    f"tool calls, got {len(actual_tool_calls)}"
                )

            for index, expected in enumerate(self.expected_tool_calls):
                if index >= len(actual_tool_calls):
                    break

                actual = actual_tool_calls[index]

                expected_tool = expected.get("tool")
                actual_tool = actual.get("tool")

                if actual_tool != expected_tool:
                    violations.append(
                        f"Tool call #{index + 1}: expected "
                        f"'{expected_tool}', got '{actual_tool}'"
                    )
                    continue

                expected_args = expected.get("args", {})
                actual_args = actual.get("args", {})

                for key, expected_value in expected_args.items():
                    actual_value = actual_args.get(key)

                    if actual_value != expected_value:
                        violations.append(
                            f"Tool '{expected_tool}' argument '{key}': "
                            f"expected '{expected_value}', "
                            f"got '{actual_value}'"
                        )

        # ---------------------------------------------------------
        # Find assistant reply
        # ---------------------------------------------------------
        assistant_messages = [
            str(step.content)
            for step in trajectory.steps
            if step.type == StepType.MESSAGE
            and step.metadata.get("role") == "assistant"
        ]

        final_response = (
            assistant_messages[-1]
            if assistant_messages
            else ""
        )

        # ---------------------------------------------------------
        # Validate expected response content
        # ---------------------------------------------------------
        for expected_text in self.final_response_contains:
            if expected_text.lower() not in final_response.lower():
                violations.append(
                    f"Final response does not contain: '{expected_text}'"
                )

        verdict = Verdict.FAIL if violations else Verdict.PASS

        return EvalResult(
            name=self.name,
            verdict=verdict,
            reason=(
                "; ".join(violations)
                if violations
                else "Trajectory matched expected behavior"
            ),
            details={
                "actual_tool_calls": actual_tool_calls,
                "final_response": final_response,
                "violations": violations,
            },
        )
