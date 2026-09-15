
"""Validates a Trajectory against a Contract."""

from __future__ import annotations

from agentguard.contracts.schema import Contract
from agentguard.core.result import EvalResult, Verdict
from agentguard.core.trajectory import StepType, Trajectory


class ContractValidator:
    """Checks whether a trajectory complied with a given contract."""

    name = "contract_validator"

    def validate(self, trajectory: Trajectory, contract: Contract) -> EvalResult:
        violations: list[str] = []

        called_tools = [
            step.content.get("tool") if isinstance(step.content, dict) else str(step.content)
            for step in trajectory.tool_calls()
        ]

        # ---------------------------------------------------------
        # Allowed tools
        # ---------------------------------------------------------
        if contract.allowed_tools:
            for tool in called_tools:
                if tool not in contract.allowed_tools:
                    violations.append(f"Used disallowed tool: {tool}")

        # ---------------------------------------------------------
        # Forbidden tools
        # ---------------------------------------------------------
        for tool in called_tools:
            if tool in contract.forbidden_tools:
                violations.append(f"Used forbidden tool: {tool}")

        # ---------------------------------------------------------
        # Tool constraints
        # ---------------------------------------------------------
        for constraint in contract.tool_constraints:
            count = called_tools.count(constraint.tool_name)

            if constraint.required and count == 0:
                violations.append(f"Required tool never called: {constraint.tool_name}")

            if constraint.max_calls is not None and count > constraint.max_calls:
                violations.append(
                    f"Tool '{constraint.tool_name}' called {count} times "
                    f"(max {constraint.max_calls})"
                )

        # ---------------------------------------------------------
        # Required steps
        #
        # required_steps is an ordered sequence.
        #
        # Example:
        #   ["verify_identity", "issue_refund"]
        #
        # means verify_identity must happen before issue_refund.
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
                violations.append("Required steps not completed in order: " + " -> ".join(missing))

        # ---------------------------------------------------------
        # Max turns
        # ---------------------------------------------------------
        if contract.max_turns is not None:
            n_messages = sum(1 for s in trajectory.steps if s.type == StepType.MESSAGE)

            if n_messages > contract.max_turns:
                violations.append(f"Exceeded max_turns: {n_messages} > {contract.max_turns}")

        verdict = Verdict.FAIL if violations else Verdict.PASS

        return EvalResult(
            name=self.name,
            verdict=verdict,
            reason=("; ".join(violations) if violations else "Contract satisfied"),
            details={
                "violations": violations,
                "called_tools": called_tools,
            },
        )

