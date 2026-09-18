"""Trajectory-level evaluator for AI agent behavior."""

from __future__ import annotations

from typing import Any

from agentguard.core.evaluator import Evaluator
from agentguard.core.result import EvalResult, Evidence, FailureType, Verdict
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
        evidence: list[Evidence] = []

        tool_call_steps = [s for s in trajectory.steps if s.type == StepType.TOOL_CALL]
        actual_tool_calls = [s.content for s in tool_call_steps if isinstance(s.content, dict)]

        # ---------------------------------------------------------
        # Validate expected tool calls in order
        # ---------------------------------------------------------
        if self.expected_tool_calls:
            if len(actual_tool_calls) < len(self.expected_tool_calls):
                evidence.append(
                    Evidence(
                        failure_type=FailureType.PREMATURE_TERMINATION,
                        message=(
                            f"Expected at least {len(self.expected_tool_calls)} "
                            f"tool calls, got {len(actual_tool_calls)}"
                        ),
                        expected=f"at least {len(self.expected_tool_calls)} tool calls",
                        actual=f"{len(actual_tool_calls)} tool calls",
                    )
                )

            for index, expected in enumerate(self.expected_tool_calls):
                if index >= len(actual_tool_calls):
                    break

                actual = actual_tool_calls[index]
                step_index = trajectory.steps.index(tool_call_steps[index])

                expected_tool = expected.get("tool")
                actual_tool = actual.get("tool")

                if actual_tool != expected_tool:
                    evidence.append(
                        Evidence(
                            failure_type=FailureType.WRONG_TOOL,
                            step_index=step_index,
                            message=(
                                f"Tool call #{index + 1}: expected "
                                f"'{expected_tool}', got '{actual_tool}'"
                            ),
                            expected=expected_tool,
                            actual=actual_tool,
                        )
                    )
                    continue

                expected_args = expected.get("args", {})
                actual_args = actual.get("args", {})

                for key, expected_value in expected_args.items():
                    actual_value = actual_args.get(key)

                    if actual_value != expected_value:
                        evidence.append(
                            Evidence(
                                failure_type=FailureType.WRONG_ARGUMENT,
                                step_index=step_index,
                                message=(
                                    f"Tool '{expected_tool}' argument '{key}': "
                                    f"expected '{expected_value}', "
                                    f"got '{actual_value}'"
                                ),
                                expected={key: expected_value},
                                actual={key: actual_value},
                            )
                        )

        # ---------------------------------------------------------
        # Find assistant reply
        # ---------------------------------------------------------
        assistant_message_steps = [
            s
            for s in trajectory.steps
            if s.type == StepType.MESSAGE and s.metadata.get("role") == "assistant"
        ]
        final_response = str(assistant_message_steps[-1].content) if assistant_message_steps else ""

        # ---------------------------------------------------------
        # Validate expected response content
        # ---------------------------------------------------------
        for expected_text in self.final_response_contains:
            if expected_text.lower() not in final_response.lower():
                evidence.append(
                    Evidence(
                        failure_type=(
                            FailureType.INCOMPLETE_RESPONSE
                            if not final_response
                            else FailureType.WRONG_FINAL_ANSWER
                        ),
                        step_index=(
                            trajectory.steps.index(assistant_message_steps[-1])
                            if assistant_message_steps
                            else None
                        ),
                        message=f"Final response does not contain: '{expected_text}'",
                        expected=expected_text,
                        actual=final_response,
                    )
                )

        verdict = Verdict.FAIL if evidence else Verdict.PASS

        return EvalResult(
            name=self.name,
            verdict=verdict,
            reason=(
                "; ".join(e.message for e in evidence)
                if evidence
                else "Trajectory matched expected behavior"
            ),
            details={
                "actual_tool_calls": actual_tool_calls,
                "final_response": final_response,
            },
            evidence=evidence,
        )
