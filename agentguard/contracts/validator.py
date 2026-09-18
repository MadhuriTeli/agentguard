"""Validates a Trajectory against a Contract."""

from __future__ import annotations

from agentguard.contracts.schema import Contract
from agentguard.core.result import EvalResult, Evidence, FailureType, Verdict
from agentguard.core.trajectory import StepType, Trajectory


class ContractValidator:
    """Checks whether a trajectory complied with a given contract."""

    name = "contract_validator"

    def validate(self, trajectory: Trajectory, contract: Contract) -> EvalResult:
        evidence: list[Evidence] = []

        tool_call_steps = trajectory.tool_calls()
        called_tools: list[str] = [
            str(step.content.get("tool")) if isinstance(step.content, dict) else str(step.content)
            for step in tool_call_steps
        ]

        # ---------------------------------------------------------
        # Allowed tools
        # ---------------------------------------------------------
        if contract.allowed_tools:
            for index, tool in enumerate(called_tools):
                if tool not in contract.allowed_tools:
                    evidence.append(
                        Evidence(
                            failure_type=FailureType.FORBIDDEN_TOOL,
                            step_index=trajectory.steps.index(tool_call_steps[index]),
                            message=f"Used disallowed tool: {tool}",
                            expected=contract.allowed_tools,
                            actual=tool,
                        )
                    )

        # ---------------------------------------------------------
        # Forbidden tools
        # ---------------------------------------------------------
        for index, tool in enumerate(called_tools):
            if tool in contract.forbidden_tools:
                evidence.append(
                    Evidence(
                        failure_type=FailureType.FORBIDDEN_TOOL,
                        step_index=trajectory.steps.index(tool_call_steps[index]),
                        message=f"Used forbidden tool: {tool}",
                        expected=f"not in {contract.forbidden_tools}",
                        actual=tool,
                    )
                )

        # ---------------------------------------------------------
        # Tool constraints
        # ---------------------------------------------------------
        for constraint in contract.tool_constraints:
            count = called_tools.count(constraint.tool_name)

            if constraint.required and count == 0:
                evidence.append(
                    Evidence(
                        failure_type=FailureType.MISSING_REQUIRED_STEP,
                        message=f"Required tool never called: {constraint.tool_name}",
                        expected=f"{constraint.tool_name} called at least once",
                        actual="never called",
                    )
                )

            if constraint.max_calls is not None and count > constraint.max_calls:
                evidence.append(
                    Evidence(
                        failure_type=FailureType.MAX_CALLS_EXCEEDED,
                        message=(
                            f"Tool '{constraint.tool_name}' called {count} times "
                            f"(max {constraint.max_calls})"
                        ),
                        expected=f"<= {constraint.max_calls} calls",
                        actual=f"{count} calls",
                    )
                )

        # ---------------------------------------------------------
        # Required steps — an ordered sequence.
        #
        # Example: ["verify_identity", "issue_refund"] means
        # verify_identity must happen before issue_refund.
        #
        # A step that never appears at all gets MISSING_REQUIRED_STEP; a
        # step whose only occurrence(s) happened before an earlier
        # required step completed (i.e. it exists in called_tools but the
        # ordered cursor never reaches it) gets ORDERING_VIOLATION instead
        # — those are different, actionable facts for a developer.
        # ---------------------------------------------------------
        if contract.required_steps:
            cursor = 0

            for tool in called_tools:
                if (
                    cursor < len(contract.required_steps)
                    and tool == contract.required_steps[cursor]
                ):
                    cursor += 1

            if cursor < len(contract.required_steps):
                missing = contract.required_steps[cursor:]
                for step_name in missing:
                    failure_type = (
                        FailureType.ORDERING_VIOLATION
                        if step_name in called_tools
                        else FailureType.MISSING_REQUIRED_STEP
                    )
                    evidence.append(
                        Evidence(
                            failure_type=failure_type,
                            message=(
                                "Required steps not completed in order: " + " -> ".join(missing)
                            ),
                            expected=" -> ".join(contract.required_steps),
                            actual=" -> ".join(called_tools) or "(no tool calls)",
                        )
                    )
                    break  # one evidence entry for the whole missing tail is enough

        # ---------------------------------------------------------
        # Max turns
        # ---------------------------------------------------------
        if contract.max_turns is not None:
            n_messages = sum(1 for s in trajectory.steps if s.type == StepType.MESSAGE)

            if n_messages > contract.max_turns:
                evidence.append(
                    Evidence(
                        failure_type=FailureType.MAX_TURNS_EXCEEDED,
                        message=f"Exceeded max_turns: {n_messages} > {contract.max_turns}",
                        expected=f"<= {contract.max_turns} turns",
                        actual=f"{n_messages} turns",
                    )
                )

        verdict = Verdict.FAIL if evidence else Verdict.PASS
        reason = "; ".join(e.message for e in evidence) if evidence else "Contract satisfied"

        return EvalResult(
            name=self.name,
            verdict=verdict,
            reason=reason,
            details={"called_tools": called_tools},
            evidence=evidence,
        )
